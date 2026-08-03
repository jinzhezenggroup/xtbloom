#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_total_energy.cuh"
#include "model/gfn2/d4.hpp"
#include "model/gfn2/repulsion.hpp"

#define CHECK(condition)                                                              \
  do {                                                                                \
    if (!(condition)) {                                                               \
      std::cerr << "check failed at line " << __LINE__ << ": " << #condition << '\n'; \
      return __LINE__;                                                                \
    }                                                                                 \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using gpuxtb::detail::cuda::compose_gfn2_total_energy_cuda;
using gpuxtb::detail::cuda::Gfn2TotalEnergyComponent;
using gpuxtb::detail::cuda::Gfn2TotalEnergyDeviceBatch;
using gpuxtb::detail::cuda::Gfn2TotalEnergyDeviceError;
using gpuxtb::detail::cuda::Gfn2TotalEnergyDeviceInput;
using gpuxtb::detail::cuda::Gfn2TotalEnergyDeviceResults;
using gpuxtb::detail::cuda::Gfn2TotalEnergyDeviceSccState;
using gpuxtb::detail::cuda::Gfn2TotalEnergyDeviceWorkspace;
using gpuxtb::detail::cuda::reset_gfn2_total_energy_device_errors_cuda;

constexpr std::uint64_t kPlanToken = 0x67f1a1e5ULL;
constexpr double kSentinel = 12345.5;

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
    if (values.size() > count_) {
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
  std::vector<double> scc_free_energy;
  std::vector<double> repulsion;
  std::vector<double> d4_atm;
  std::vector<double> expected;
  std::vector<gpuxtb_status_t> statuses;
  std::vector<std::uint8_t> converged;
};

/*
 * Generate the two geometry-only components through the production CPU
 * implementations. The CUDA test then checks only the final composition
 * slice against that independent component source and exact host addition
 * order; repulsion and D4 kernels retain their own focused parity tests.
 */
bool make_host_case(std::size_t batch_size, bool enable_d4_atm, HostCase& result,
                    std::string& error) {
  constexpr std::size_t atoms_per_system = 4u;
  const std::size_t total_atoms = batch_size * atoms_per_system;
  std::vector<std::int64_t> atom_offsets(batch_size + 1u);
  std::vector<std::int32_t> atomic_numbers(total_atoms);
  std::vector<double> positions(total_atoms * 3u, 0.0);
  for (std::size_t system = 0; system < batch_size; ++system) {
    const std::size_t atom = system * atoms_per_system;
    atom_offsets[system] = static_cast<std::int64_t>(atom);
    atomic_numbers[atom] = 8;
    atomic_numbers[atom + 1u] = 1;
    atomic_numbers[atom + 2u] = 1;
    atomic_numbers[atom + 3u] = 6;
    const double perturbation = 1.0e-3 * static_cast<double>(system % 13u);
    positions[(atom + 1u) * 3u] = 1.43 + perturbation;
    positions[(atom + 1u) * 3u + 1u] = 1.11;
    positions[(atom + 2u) * 3u] = -1.43;
    positions[(atom + 2u) * 3u + 1u] = 1.11 - 0.5 * perturbation;
    positions[(atom + 3u) * 3u] = 0.35;
    positions[(atom + 3u) * 3u + 1u] = -0.20;
    positions[(atom + 3u) * 3u + 2u] = 2.45 + 0.25 * perturbation;
  }
  atom_offsets[batch_size] = static_cast<std::int64_t>(total_atoms);

  gpuxtb::detail::gfn2::RepulsionPlan repulsion_plan;
  if (gpuxtb::detail::gfn2::make_repulsion_plan(static_cast<std::int64_t>(batch_size),
                                                static_cast<std::int64_t>(total_atoms),
                                                atom_offsets.data(), atomic_numbers.data(),
                                                repulsion_plan, error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  result.repulsion.assign(batch_size, 0.0);
  if (gpuxtb::detail::gfn2::add_repulsion_cpu(repulsion_plan, positions.data(),
                                              result.repulsion.data(), nullptr,
                                              error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }

  result.d4_atm.assign(batch_size, 0.0);
  if (enable_d4_atm) {
    gpuxtb::detail::gfn2::D4Plan d4_plan;
    if (gpuxtb::detail::gfn2::make_d4_plan(
            static_cast<std::int64_t>(batch_size), static_cast<std::int64_t>(total_atoms),
            atom_offsets.data(), atomic_numbers.data(), d4_plan, error) != GPUXTB_STATUS_SUCCESS) {
      return false;
    }
    std::vector<std::byte> workspace_storage(d4_plan.workspace_size_bytes() +
                                             gpuxtb::detail::gfn2::kD4WorkspaceAlignment - 1u);
    const std::uintptr_t unaligned = reinterpret_cast<std::uintptr_t>(workspace_storage.data());
    const std::uintptr_t aligned = (unaligned + gpuxtb::detail::gfn2::kD4WorkspaceAlignment - 1u) &
                                   ~(gpuxtb::detail::gfn2::kD4WorkspaceAlignment - 1u);
    gpuxtb::detail::gfn2::D4Workspace d4_workspace;
    if (gpuxtb::detail::gfn2::bind_d4_workspace(d4_plan, reinterpret_cast<void*>(aligned),
                                                d4_plan.workspace_size_bytes(), d4_workspace,
                                                error) != GPUXTB_STATUS_SUCCESS) {
      return false;
    }
    std::vector<double> pair_data(static_cast<std::size_t>(d4_plan.total_pairs()) *
                                  gpuxtb::detail::gfn2::kD4PairDataElements);
    std::vector<double> coordination(total_atoms);
    gpuxtb::detail::gfn2::D4GeometryCache cache;
    if (gpuxtb::detail::gfn2::update_d4_geometry_cache_cpu(
            d4_plan, positions.data(), 1u, pair_data.data(), pair_data.size(), coordination.data(),
            coordination.size(), d4_workspace, cache, error) != GPUXTB_STATUS_SUCCESS ||
        gpuxtb::detail::gfn2::evaluate_d4_atm_cpu(d4_plan, cache, result.d4_atm.data(),
                                                  d4_workspace, error) != GPUXTB_STATUS_SUCCESS) {
      return false;
    }
  }

  result.scc_free_energy.resize(batch_size);
  result.expected.resize(batch_size);
  result.statuses.assign(batch_size, GPUXTB_STATUS_SUCCESS);
  result.converged.assign(batch_size, 1u);
  for (std::size_t system = 0; system < batch_size; ++system) {
    result.scc_free_energy[system] =
        -15.0 - 0.03125 * static_cast<double>(system) + 1.0e-6 * static_cast<double>(system % 7u);
    double total = result.scc_free_energy[system];
    total += result.repulsion[system];
    if (enable_d4_atm) {
      total += result.d4_atm[system];
    }
    result.expected[system] = total;
  }
  error.clear();
  return true;
}

struct DeviceCase {
  explicit DeviceCase(std::size_t batch_size, bool enable_d4_atm)
      : batch{static_cast<std::int64_t>(batch_size),
              enable_d4_atm ? static_cast<std::uint32_t>(Gfn2TotalEnergyComponent::kD4Atm) : 0u,
              kPlanToken} {
    require(scc_free_energy.allocate(batch_size));
    require(repulsion.allocate(batch_size));
    require(d4_atm.allocate(enable_d4_atm ? batch_size : 0u));
    require(statuses.allocate(batch_size));
    require(converged.allocate(batch_size));
    require(total_energy.allocate(batch_size));
    require(system_errors.allocate(batch_size));
    require(device_error.allocate(1u));
    require(sequence_active.allocate(1u));
    input = {scc_free_energy.get(),
             static_cast<std::int64_t>(batch_size),
             repulsion.get(),
             static_cast<std::int64_t>(batch_size),
             d4_atm.get(),
             enable_d4_atm ? static_cast<std::int64_t>(batch_size) : 0,
             kPlanToken};
    state = {statuses.get(), converged.get(), static_cast<std::int64_t>(batch_size), kPlanToken};
    results = {total_energy.get(), static_cast<std::int64_t>(batch_size), kPlanToken};
    workspace = {sequence_active.get(), 1, kPlanToken};
  }

  cudaError_t upload(const HostCase& host, cudaStream_t stream) {
    cudaError_t status = scc_free_energy.upload(host.scc_free_energy, stream);
    if (status != cudaSuccess) return status;
    status = repulsion.upload(host.repulsion, stream);
    if (status != cudaSuccess) return status;
    if (d4_atm.size() != 0u) {
      status = d4_atm.upload(host.d4_atm, stream);
      if (status != cudaSuccess) return status;
    }
    status = statuses.upload(host.statuses, stream);
    if (status != cudaSuccess) return status;
    status = converged.upload(host.converged, stream);
    if (status != cudaSuccess) return status;
    return total_energy.upload(std::vector<double>(host.expected.size(), kSentinel), stream);
  }

  Gfn2TotalEnergyDeviceBatch batch{};
  Gfn2TotalEnergyDeviceInput input{};
  Gfn2TotalEnergyDeviceSccState state{};
  Gfn2TotalEnergyDeviceResults results{};
  Gfn2TotalEnergyDeviceWorkspace workspace{};
  DeviceBuffer<double> scc_free_energy;
  DeviceBuffer<double> repulsion;
  DeviceBuffer<double> d4_atm;
  DeviceBuffer<gpuxtb_status_t> statuses;
  DeviceBuffer<std::uint8_t> converged;
  DeviceBuffer<double> total_energy;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;
  DeviceBuffer<std::uint32_t> sequence_active;

 private:
  static void require(cudaError_t status) {
    if (status != cudaSuccess) {
      std::cerr << "CUDA fixture setup failed: " << cudaGetErrorString(status) << '\n';
      std::abort();
    }
  }
};

int run_case(std::size_t batch_size, bool enable_d4_atm) {
  HostCase host;
  std::string error;
  CHECK(make_host_case(batch_size, enable_d4_atm, host, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceCase device(batch_size, enable_d4_atm);
  CUDA_CHECK(device.upload(host, stream));
  CUDA_CHECK(reset_gfn2_total_energy_device_errors_cuda(
      device.batch.batch_size, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(compose_gfn2_total_energy_cuda(
      device.batch, device.input, device.state, device.results, device.workspace,
      device.system_errors.get(), device.device_error.get(), stream));
  std::vector<double> actual;
  std::vector<std::uint32_t> system_errors;
  std::vector<std::uint32_t> device_error;
  CUDA_CHECK(device.total_energy.download(actual, stream));
  CUDA_CHECK(device.system_errors.download(system_errors, stream));
  CUDA_CHECK(device.device_error.download(device_error, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CHECK(device_error == std::vector<std::uint32_t>{0u});
  CHECK(system_errors == std::vector<std::uint32_t>(batch_size, 0u));
  CHECK(actual == host.expected);
  return 0;
}

int test_cpu_parity_batch_matrix() {
  for (const std::size_t batch_size : std::array<std::size_t, 4>{1u, 8u, 32u, 128u}) {
    CHECK(run_case(batch_size, true) == 0);
  }
  /* A disabled ATM descriptor must be canonical null/zero and skips the add. */
  CHECK(run_case(8u, false) == 0);
  return 0;
}

int test_terminal_gating_and_peer_failures() {
  constexpr std::size_t batch_size = 8u;
  HostCase host;
  std::string error;
  CHECK(make_host_case(batch_size, true, host, error));
  host.converged[1] = 0u;
  host.scc_free_energy[1] = std::numeric_limits<double>::quiet_NaN();
  host.converged[2] = 2u;
  host.statuses[3] = GPUXTB_STATUS_SCC_NOT_CONVERGED;
  host.scc_free_energy[4] = std::numeric_limits<double>::infinity();
  host.repulsion[5] = std::numeric_limits<double>::quiet_NaN();
  host.d4_atm[6] = -std::numeric_limits<double>::infinity();
  host.scc_free_energy[7] = std::numeric_limits<double>::max();
  host.repulsion[7] = std::numeric_limits<double>::max();

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceCase device(batch_size, true);
  CUDA_CHECK(device.upload(host, stream));
  CUDA_CHECK(reset_gfn2_total_energy_device_errors_cuda(
      device.batch.batch_size, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(compose_gfn2_total_energy_cuda(
      device.batch, device.input, device.state, device.results, device.workspace,
      device.system_errors.get(), device.device_error.get(), stream));
  std::vector<double> actual;
  std::vector<std::uint32_t> errors;
  std::vector<std::uint32_t> sticky;
  CUDA_CHECK(device.total_energy.download(actual, stream));
  CUDA_CHECK(device.system_errors.download(errors, stream));
  CUDA_CHECK(device.device_error.download(sticky, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));

  CHECK(actual[0] == host.expected[0]);
  for (std::size_t system = 1; system < batch_size; ++system) {
    CHECK(actual[system] == kSentinel);
  }
  CHECK(errors[0] == 0u);
  CHECK(errors[1] == 0u);  // Inactive peers are not numerically inspected.
  CHECK(errors[2] ==
        static_cast<std::uint32_t>(Gfn2TotalEnergyDeviceError::kInvalidConvergenceFlag));
  CHECK(errors[3] ==
        static_cast<std::uint32_t>(Gfn2TotalEnergyDeviceError::kInconsistentSccStatus));
  CHECK(errors[4] ==
        static_cast<std::uint32_t>(Gfn2TotalEnergyDeviceError::kNonfiniteSccFreeEnergy));
  CHECK(errors[5] == static_cast<std::uint32_t>(Gfn2TotalEnergyDeviceError::kNonfiniteRepulsion));
  CHECK(errors[6] == static_cast<std::uint32_t>(Gfn2TotalEnergyDeviceError::kNonfiniteD4Atm));
  CHECK(errors[7] ==
        static_cast<std::uint32_t>(Gfn2TotalEnergyDeviceError::kNonfiniteSccRepulsionSum));
  CHECK(sticky.size() == 1u && sticky[0] != 0u);
  return 0;
}

int test_host_validation() {
  DeviceCase device(1u, true);
  Gfn2TotalEnergyDeviceBatch invalid = device.batch;
  invalid.enabled_components = 0x80000000u;
  CHECK(compose_gfn2_total_energy_cuda(invalid, device.input, device.state, device.results,
                                       device.workspace, device.system_errors.get(),
                                       device.device_error.get()) == cudaErrorInvalidValue);
  Gfn2TotalEnergyDeviceResults alias = device.results;
  alias.total_energy = device.scc_free_energy.get();
  CHECK(compose_gfn2_total_energy_cuda(device.batch, device.input, device.state, alias,
                                       device.workspace, device.system_errors.get(),
                                       device.device_error.get()) == cudaErrorInvalidValue);
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status == cudaErrorNoDevice || count_status == cudaErrorInsufficientDriver ||
      device_count == 0) {
    cudaGetLastError();
    std::cout << "CUDA GFN2 total-energy test skipped: no CUDA device\n";
    return 0;
  }
  CUDA_CHECK(count_status);
  CHECK(test_cpu_parity_batch_matrix() == 0);
  CHECK(test_terminal_gating_and_peer_failures() == 0);
  CHECK(test_host_validation() == 0);
  std::cout << "CUDA GFN2 total-energy tests passed\n";
  return 0;
}
