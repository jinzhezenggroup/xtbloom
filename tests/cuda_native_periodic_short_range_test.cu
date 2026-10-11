// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_native_periodic_short_range.cuh"
#include "backends/cuda/periodic_topology.cuh"
#include "model/gfn2/coordination.hpp"
#include "model/gfn2/periodic_topology.hpp"
#include "model/gfn2/repulsion.hpp"
#include "runtime/backend.hpp"

#define CHECK(condition)                                                                        \
  do {                                                                                          \
    if (!(condition)) {                                                                         \
      std::cerr << "CUDA native periodic short-range check failed at line " << __LINE__ << ": " \
                << #condition << '\n';                                                          \
      return __LINE__;                                                                          \
    }                                                                                           \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using xtbloom::detail::cuda::Gfn2CudaPeriodicTopology;
using xtbloom::detail::cuda::Gfn2CudaPeriodicTopologyInput;
using xtbloom::detail::cuda::Gfn2NativePeriodicShortRangeDeviceBatch;
using xtbloom::detail::cuda::Gfn2NativePeriodicShortRangeDeviceWorkspace;
using xtbloom::detail::gfn2::CoordinationPlan;
using xtbloom::detail::gfn2::PeriodicShortRangeGeometry;
using xtbloom::detail::gfn2::PeriodicShortRangePlan;
using xtbloom::detail::gfn2::PeriodicShortRangeWorkspace;
using xtbloom::detail::gfn2::RepulsionPlan;

constexpr std::uint64_t kPlanToken = 0x4e41544956455053ULL;
constexpr double kSentinel = -913.25;

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (data_ != nullptr) (void)cudaFree(data_);
  }

  cudaError_t allocate(std::size_t count) {
    count_ = count;
    return cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
  }

  cudaError_t upload(const std::vector<T>& values, cudaStream_t stream) {
    if (values.size() > count_) return cudaErrorInvalidValue;
    return cudaMemcpyAsync(data_, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice,
                           stream);
  }

  cudaError_t download(std::vector<T>& values, cudaStream_t stream) const {
    if (values.size() > count_) return cudaErrorInvalidValue;
    return cudaMemcpyAsync(values.data(), data_, values.size() * sizeof(T), cudaMemcpyDeviceToHost,
                           stream);
  }

  T* get() noexcept { return data_; }
  const T* get() const noexcept { return data_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

class AlignedStorage {
 public:
  AlignedStorage() = default;
  AlignedStorage(const AlignedStorage&) = delete;
  AlignedStorage& operator=(const AlignedStorage&) = delete;
  ~AlignedStorage() { std::free(pointer_); }

  bool allocate(std::size_t bytes) {
    const std::size_t rounded = (bytes + 63u) / 64u * 64u;
    pointer_ = std::aligned_alloc(64u, rounded);
    return pointer_ != nullptr;
  }

  void* get() const noexcept { return pointer_; }

 private:
  void* pointer_ = nullptr;
};

bool close(double actual, double expected, double tolerance = 2.0e-10) {
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::abs(actual - expected) <= tolerance * (1.0 + std::abs(expected));
}

int test_periodic_short_range_parity_and_peer_isolation(cudaStream_t stream, int device) {
  /* H/O/H exercises the fitted H/H light-element branch as well as the
   * ordinary GFN2 repulsion branch. The second peer is deliberately poisoned
   * after the CPU reference is built to prove peer-local transactional output.
   */
  const std::vector<std::int64_t> atom_offsets{0, 3, 6};
  const std::vector<std::int32_t> valid_atomic_numbers{1, 8, 1, 1, 8, 1};
  const std::vector<std::int32_t> invalid_atomic_numbers{1, 8, 1, 0, 8, 1};
  /* Keep this deliberately skewed cell and the out-of-cell coordinates below:
   * the CPU/GPU parity checks are also the wrapping regression.  A Cartesian
   * component-wise modulo would disagree with the full inverse/forward lattice
   * transform for this fixture, even though an orthogonal cell would pass. */
  const std::vector<double> cells{
      11.7, 0.0, 0.0, 1.1, 12.9, 0.0, -0.7, 0.8, 14.3,
      11.7, 0.0, 0.0, 1.1, 12.9, 0.0, -0.7, 0.8, 14.3,
  };
  const std::vector<double> positions{
      12.10, 13.20, 14.30, -10.45, 0.20, 0.30, 0.10, 1.75, 0.30,
      0.40,  0.30,  0.50,  1.85,   0.30, 0.50, 0.40, 1.80, 0.50,
  };

  std::string error;
  CoordinationPlan coordination_plan;
  CHECK(xtbloom::detail::gfn2::make_coordination_plan(
            2, 6, atom_offsets.data(), valid_atomic_numbers.data(), coordination_plan, error) ==
        XTBLOOM_STATUS_SUCCESS);
  RepulsionPlan repulsion_plan;
  CHECK(xtbloom::detail::gfn2::make_repulsion_plan(2, 6, atom_offsets.data(),
                                                   valid_atomic_numbers.data(), repulsion_plan,
                                                   error) == XTBLOOM_STATUS_SUCCESS);
  PeriodicShortRangePlan periodic_plan;
  CHECK(xtbloom::detail::gfn2::make_periodic_short_range_plan(2, 6, atom_offsets.data(),
                                                              cells.data(), periodic_plan,
                                                              error) == XTBLOOM_STATUS_SUCCESS);
  AlignedStorage host_workspace;
  CHECK(host_workspace.allocate(periodic_plan.workspace_size_bytes()));
  PeriodicShortRangeWorkspace host_workspace_view;
  CHECK(xtbloom::detail::gfn2::bind_periodic_short_range_workspace(
            periodic_plan, host_workspace.get(), periodic_plan.workspace_size_bytes(),
            host_workspace_view, error) == XTBLOOM_STATUS_SUCCESS);
  PeriodicShortRangeGeometry geometry;
  CHECK(xtbloom::detail::gfn2::update_periodic_short_range_geometry_cpu(
            periodic_plan, positions.data(), 1u, host_workspace_view, geometry, error) ==
        XTBLOOM_STATUS_SUCCESS);

  std::vector<double> cpu_coordination(6);
  std::vector<double> cpu_repulsion_energy(6);
  std::vector<double> cpu_gradients(18);
  std::vector<double> cpu_strain(18);
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_coordination_cpu(
            coordination_plan, periodic_plan, geometry, cpu_coordination.data(),
            host_workspace_view, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_repulsion_cpu(
            repulsion_plan, periodic_plan, geometry, cpu_repulsion_energy.data(),
            cpu_gradients.data(), cpu_strain.data(), host_workspace_view,
            error) == XTBLOOM_STATUS_SUCCESS);

  CHECK(xtbloom::detail::ensure_cuda_gfn2_parameters(device, error));
  Gfn2CudaPeriodicTopology topology;
  const Gfn2CudaPeriodicTopologyInput topology_input{
      2, 6, atom_offsets.data(), cells.data(),
      /* periodic_axes is implicit in this topology owner: native XYZ */
      nullptr, 25.0, kPlanToken, 1u};
  /* The owner requires an explicit mask array; keep the assignment separate
   * so the aggregate remains easy to audit against the public ABI. */
  const std::vector<std::int32_t> periodic_axes{XTBLOOM_PERIODIC_AXES_XYZ,
                                                XTBLOOM_PERIODIC_AXES_XYZ};
  Gfn2CudaPeriodicTopologyInput input = topology_input;
  input.periodic_axes = periodic_axes.data();
  CHECK(Gfn2CudaPeriodicTopology::create(input, stream, topology).success());

  DeviceBuffer<std::int32_t> atomic_numbers;
  DeviceBuffer<double> device_positions;
  DeviceBuffer<double> covalent_radii;
  DeviceBuffer<double> wrapped_positions;
  DeviceBuffer<double> coordination_scratch;
  DeviceBuffer<double> repulsion_energy_scratch;
  DeviceBuffer<double> repulsion_gradient_scratch;
  DeviceBuffer<double> repulsion_strain_scratch;
  DeviceBuffer<double> coordination;
  DeviceBuffer<double> repulsion_energy;
  DeviceBuffer<double> gradients;
  DeviceBuffer<double> strain;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;
  CHECK(atomic_numbers.allocate(6) == cudaSuccess);
  CHECK(device_positions.allocate(18) == cudaSuccess);
  CHECK(covalent_radii.allocate(6) == cudaSuccess);
  CHECK(wrapped_positions.allocate(18) == cudaSuccess);
  CHECK(coordination_scratch.allocate(6) == cudaSuccess);
  CHECK(repulsion_energy_scratch.allocate(2) == cudaSuccess);
  CHECK(repulsion_gradient_scratch.allocate(18) == cudaSuccess);
  CHECK(repulsion_strain_scratch.allocate(18) == cudaSuccess);
  CHECK(coordination.allocate(6) == cudaSuccess);
  CHECK(repulsion_energy.allocate(2) == cudaSuccess);
  CHECK(gradients.allocate(18) == cudaSuccess);
  CHECK(strain.allocate(18) == cudaSuccess);
  CHECK(system_errors.allocate(2) == cudaSuccess);
  CHECK(device_error.allocate(1) == cudaSuccess);
  CHECK(atomic_numbers.upload(invalid_atomic_numbers, stream) == cudaSuccess);
  CHECK(device_positions.upload(positions, stream) == cudaSuccess);
  CHECK(covalent_radii.upload(coordination_plan.covalent_radius, stream) == cudaSuccess);
  const std::vector<double> cn_seed(6, kSentinel);
  const std::vector<double> energy_seed(2, kSentinel);
  const std::vector<double> gradient_seed(18, kSentinel);
  const std::vector<double> strain_seed(18, kSentinel);
  CHECK(coordination.upload(cn_seed, stream) == cudaSuccess);
  CHECK(repulsion_energy.upload(energy_seed, stream) == cudaSuccess);
  CHECK(gradients.upload(gradient_seed, stream) == cudaSuccess);
  CHECK(strain.upload(strain_seed, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);

  const auto topology_view = topology.device_view();
  const Gfn2NativePeriodicShortRangeDeviceBatch batch{
      topology_view, atomic_numbers.get(), 6, device_positions.get(), 18, covalent_radii.get(), 6};
  const Gfn2NativePeriodicShortRangeDeviceWorkspace workspace{wrapped_positions.get(),
                                                              18,
                                                              coordination_scratch.get(),
                                                              6,
                                                              repulsion_energy_scratch.get(),
                                                              2,
                                                              repulsion_gradient_scratch.get(),
                                                              18,
                                                              repulsion_strain_scratch.get(),
                                                              18,
                                                              topology.plan_token()};
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_native_periodic_short_range_errors_cuda(
      2, system_errors.get(), device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_native_periodic_short_range_cuda(
      batch, workspace, coordination.get(), repulsion_energy.get(), gradients.get(), strain.get(),
      system_errors.get(), device_error.get(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<double> gpu_coordination(6);
  std::vector<double> gpu_energy(2);
  std::vector<double> gpu_gradients(18);
  std::vector<double> gpu_strain(18);
  std::vector<std::uint32_t> gpu_system_errors(2);
  std::vector<std::uint32_t> gpu_device_error(1);
  CHECK(coordination.download(gpu_coordination, stream) == cudaSuccess);
  CHECK(repulsion_energy.download(gpu_energy, stream) == cudaSuccess);
  CHECK(gradients.download(gpu_gradients, stream) == cudaSuccess);
  CHECK(strain.download(gpu_strain, stream) == cudaSuccess);
  CHECK(system_errors.download(gpu_system_errors, stream) == cudaSuccess);
  CHECK(device_error.download(gpu_device_error, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);

  for (std::size_t atom = 0; atom < 3u; ++atom) {
    CHECK(close(gpu_coordination[atom], cpu_coordination[atom]));
    for (int axis = 0; axis < 3; ++axis) {
      CHECK(close(gpu_gradients[atom * 3u + axis], cpu_gradients[atom * 3u + axis]));
    }
  }
  const double expected_energy =
      std::accumulate(cpu_repulsion_energy.begin(), cpu_repulsion_energy.begin() + 3u, 0.0);
  CHECK(close(gpu_energy[0], expected_energy));
  for (std::size_t component = 0; component < 9u; ++component) {
    CHECK(close(gpu_strain[component], cpu_strain[component]));
  }
  CHECK(gpu_system_errors[0] == 0u);
  CHECK(gpu_system_errors[1] != 0u);
  CHECK(gpu_device_error[0] == 0u || gpu_device_error[0] == 7u);
  for (std::size_t atom = 3u; atom < 6u; ++atom) CHECK(gpu_coordination[atom] == kSentinel);
  CHECK(gpu_energy[1] == kSentinel);
  for (std::size_t index = 9u; index < 18u; ++index) {
    CHECK(gpu_gradients[index] == kSentinel);
    CHECK(gpu_strain[index] == kSentinel);
  }

  /* Structural rejection happens before enqueue and does not touch outputs. */
  Gfn2NativePeriodicShortRangeDeviceBatch malformed = batch;
  malformed.atomic_number_elements = 5;
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_native_periodic_short_range_cuda(
            malformed, workspace, coordination.get(), repulsion_energy.get(), gradients.get(),
            strain.get(), system_errors.get(), device_error.get(),
            stream) == cudaErrorInvalidValue);
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_native_periodic_short_range_cuda(
            batch, workspace, coordination.get(), repulsion_energy.get(),
            reinterpret_cast<double*>(system_errors.get()), strain.get(), system_errors.get(),
            device_error.get(), stream) == cudaErrorInvalidValue);
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    std::cout << "CUDA native periodic short-range test skipped: "
              << (count_status == cudaSuccess ? "no visible device"
                                              : cudaGetErrorString(count_status))
              << '\n';
    return 0;
  }
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) return 1;
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return 1;
  const int status = test_periodic_short_range_parity_and_peer_isolation(stream, device);
  (void)cudaStreamDestroy(stream);
  return status;
}
