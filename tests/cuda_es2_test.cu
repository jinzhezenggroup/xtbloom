#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_es2.cuh"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/es2.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

using gpuxtb::detail::cuda::Gfn2ES2DeviceBatch;
using gpuxtb::detail::cuda::Gfn2ES2DeviceCache;
using gpuxtb::detail::cuda::Gfn2ES2DeviceError;
using gpuxtb::detail::cuda::Gfn2ES2DeviceWorkspace;
using gpuxtb::detail::gfn2::BasisPlan;
using gpuxtb::detail::gfn2::ES2GeometryCache;
using gpuxtb::detail::gfn2::ES2Plan;
using gpuxtb::detail::gfn2::ES2Workspace;

constexpr std::uint64_t kGeneration = 17u;

static_assert(std::is_trivially_copyable_v<Gfn2ES2DeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2ES2DeviceCache>);
static_assert(std::is_trivially_copyable_v<Gfn2ES2DeviceWorkspace>);

bool near(double actual, double expected, double tolerance = 3.0e-13) {
  return std::abs(actual - expected) <=
         tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { (void)allocate(count); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0u)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0u);
    }
    return *this;
  }
  ~DeviceBuffer() { release(); }

  bool allocate(std::size_t count) {
    release();
    if (count == 0u) {
      return false;
    }
    count_ = count;
    if (cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)) != cudaSuccess) {
      data_ = nullptr;
      count_ = 0u;
      return false;
    }
    return true;
  }

  bool copy_from(const T* source, std::size_t count, cudaStream_t stream = nullptr) {
    return source != nullptr && count <= count_ &&
           cudaMemcpyAsync(data_, source, count * sizeof(T), cudaMemcpyHostToDevice, stream) ==
               cudaSuccess;
  }

  bool copy_to(T* target, std::size_t count, cudaStream_t stream = nullptr) const {
    return target != nullptr && count <= count_ &&
           cudaMemcpyAsync(target, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream) ==
               cudaSuccess;
  }

  bool zero(cudaStream_t stream = nullptr) {
    return data_ != nullptr && cudaMemsetAsync(data_, 0, count_ * sizeof(T), stream) == cudaSuccess;
  }

  T* get() { return data_; }
  const T* get() const { return data_; }
  std::size_t size() const { return count_; }

 private:
  void release() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
    data_ = nullptr;
    count_ = 0u;
  }

  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

struct HostEvaluation {
  BasisPlan basis;
  ES2Plan plan;
  std::vector<double> positions;
  std::vector<double> charges;
  std::vector<double> matrix;
  std::vector<double> potential;
  std::vector<double> energies;
  std::vector<double> gradients;
  std::vector<double> matrix_scratch;
  std::vector<double> shell_scratch;
  std::vector<double> batch_scratch;
  std::vector<double> gradient_scratch;
  ES2Workspace workspace;
  ES2GeometryCache cache;
};

bool make_host_evaluation(const std::vector<std::int64_t>& atom_offsets,
                          const std::vector<std::int32_t>& atomic_numbers,
                          const std::vector<double>& positions, const std::vector<double>& charges,
                          HostEvaluation& host, std::string& error) {
  const std::int64_t batch_size = static_cast<std::int64_t>(atom_offsets.size() - 1u);
  if (gpuxtb::detail::gfn2::make_basis_plan(
          batch_size, static_cast<std::int64_t>(atomic_numbers.size()), atom_offsets.data(),
          atomic_numbers.data(), host.basis, error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::make_es2_plan(host.basis, atomic_numbers.data(), host.plan, error) !=
          GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  if (positions.size() != static_cast<std::size_t>(host.plan.total_atoms() * 3) ||
      charges.size() != static_cast<std::size_t>(host.plan.total_shells())) {
    return false;
  }
  host.positions = positions;
  host.charges = charges;
  host.matrix.resize(static_cast<std::size_t>(host.plan.total_matrix_elements()));
  host.potential.resize(static_cast<std::size_t>(host.plan.total_shells()));
  host.energies.resize(static_cast<std::size_t>(host.plan.batch_size()));
  host.gradients.resize(static_cast<std::size_t>(host.plan.total_atoms() * 3));
  host.matrix_scratch.resize(host.matrix.size());
  host.shell_scratch.resize(host.potential.size());
  host.batch_scratch.resize(host.energies.size());
  host.gradient_scratch.resize(host.gradients.size());
  host.workspace = {host.matrix_scratch.data(),   host.plan.total_matrix_elements(),
                    host.shell_scratch.data(),    host.plan.total_shells(),
                    host.batch_scratch.data(),    host.plan.batch_size(),
                    host.gradient_scratch.data(), host.plan.total_atoms() * 3};
  return true;
}

bool evaluate_cpu(HostEvaluation& host, std::uint64_t generation, std::string& error) {
  std::fill(host.energies.begin(), host.energies.end(), 0.0);
  std::fill(host.gradients.begin(), host.gradients.end(), 0.0);
  return gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
             host.plan, host.positions.data(), generation, host.matrix.data(), host.matrix.size(),
             host.workspace, host.cache, error) == GPUXTB_STATUS_SUCCESS &&
         gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
             host.plan, host.cache, host.charges.data(), host.potential.data(), host.workspace,
             error) == GPUXTB_STATUS_SUCCESS &&
         gpuxtb::detail::gfn2::add_es2_energy_cpu(host.plan, host.cache, host.charges.data(),
                                                  host.energies.data(), host.workspace,
                                                  error) == GPUXTB_STATUS_SUCCESS &&
         gpuxtb::detail::gfn2::add_es2_gradient_cpu(
             host.plan, host.cache, host.positions.data(), generation, host.charges.data(),
             host.gradients.data(), host.workspace, error) == GPUXTB_STATUS_SUCCESS;
}

struct UploadedES2 {
  cudaStream_t stream = nullptr;
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> batch_shell_offsets;
  DeviceBuffer<std::int64_t> atom_shell_offsets;
  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<std::int64_t> shell_to_atom;
  DeviceBuffer<double> shell_hardness;
  DeviceBuffer<double> positions;
  DeviceBuffer<double> charges;
  DeviceBuffer<double> matrix;
  DeviceBuffer<double> potential;
  DeviceBuffer<double> energies;
  DeviceBuffer<double> gradients;
  DeviceBuffer<double> matrix_scratch;
  DeviceBuffer<double> shell_scratch;
  DeviceBuffer<double> batch_scratch;
  DeviceBuffer<double> gradient_scratch;
  DeviceBuffer<std::uint32_t> error;
  Gfn2ES2DeviceBatch batch;
  Gfn2ES2DeviceCache cache;
  Gfn2ES2DeviceWorkspace workspace;

  UploadedES2() = default;
  UploadedES2(const UploadedES2&) = delete;
  UploadedES2& operator=(const UploadedES2&) = delete;
  ~UploadedES2() {
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
    }
  }

  bool initialize(const HostEvaluation& host) {
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
        !atom_offsets.allocate(host.plan.atom_offsets().size()) ||
        !batch_shell_offsets.allocate(host.plan.batch_shell_offsets().size()) ||
        !atom_shell_offsets.allocate(host.plan.atom_shell_offsets().size()) ||
        !matrix_offsets.allocate(host.plan.matrix_offsets().size()) ||
        !shell_to_atom.allocate(host.plan.shell_to_atom().size()) ||
        !shell_hardness.allocate(host.plan.shell_hardness().size()) ||
        !positions.allocate(host.positions.size()) || !charges.allocate(host.charges.size()) ||
        !matrix.allocate(host.matrix.size()) || !potential.allocate(host.potential.size()) ||
        !energies.allocate(host.energies.size()) || !gradients.allocate(host.gradients.size()) ||
        !matrix_scratch.allocate(host.matrix.size()) ||
        !shell_scratch.allocate(host.potential.size()) ||
        !batch_scratch.allocate(host.energies.size()) ||
        !gradient_scratch.allocate(host.gradients.size()) || !error.allocate(1u)) {
      return false;
    }
    if (!atom_offsets.copy_from(host.plan.atom_offsets().data(), host.plan.atom_offsets().size(),
                                stream) ||
        !batch_shell_offsets.copy_from(host.plan.batch_shell_offsets().data(),
                                       host.plan.batch_shell_offsets().size(), stream) ||
        !atom_shell_offsets.copy_from(host.plan.atom_shell_offsets().data(),
                                      host.plan.atom_shell_offsets().size(), stream) ||
        !matrix_offsets.copy_from(host.plan.matrix_offsets().data(),
                                  host.plan.matrix_offsets().size(), stream) ||
        !shell_to_atom.copy_from(host.plan.shell_to_atom().data(), host.plan.shell_to_atom().size(),
                                 stream) ||
        !shell_hardness.copy_from(host.plan.shell_hardness().data(),
                                  host.plan.shell_hardness().size(), stream) ||
        !positions.copy_from(host.positions.data(), host.positions.size(), stream) ||
        !charges.copy_from(host.charges.data(), host.charges.size(), stream) ||
        !matrix.zero(stream) || !potential.zero(stream) || !energies.zero(stream) ||
        !gradients.zero(stream) || !matrix_scratch.zero(stream) || !shell_scratch.zero(stream) ||
        !batch_scratch.zero(stream) || !gradient_scratch.zero(stream) || !error.zero(stream) ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return false;
    }
    const std::uint64_t token =
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(host.plan.identity()));
    batch = {host.plan.batch_size(),
             host.plan.total_atoms(),
             host.plan.total_shells(),
             host.plan.total_matrix_elements(),
             token,
             static_cast<std::int64_t>(host.plan.atom_offsets().size()),
             static_cast<std::int64_t>(host.plan.batch_shell_offsets().size()),
             static_cast<std::int64_t>(host.plan.atom_shell_offsets().size()),
             static_cast<std::int64_t>(host.plan.matrix_offsets().size()),
             static_cast<std::int64_t>(host.plan.shell_to_atom().size()),
             static_cast<std::int64_t>(host.plan.shell_hardness().size()),
             atom_offsets.get(),
             batch_shell_offsets.get(),
             atom_shell_offsets.get(),
             matrix_offsets.get(),
             shell_to_atom.get(),
             shell_hardness.get()};
    cache = {matrix.get(), host.plan.total_matrix_elements(), kGeneration, token};
    workspace = {matrix_scratch.get(),   host.plan.total_matrix_elements(),
                 shell_scratch.get(),    host.plan.total_shells(),
                 batch_scratch.get(),    host.plan.batch_size(),
                 gradient_scratch.get(), host.plan.total_atoms() * 3};
    return token != 0u;
  }

  bool synchronize_error(std::uint32_t& value) {
    return error.copy_to(&value, 1u, stream) && cudaStreamSynchronize(stream) == cudaSuccess;
  }
};

bool reset_error(UploadedES2& device) {
  return gpuxtb::detail::cuda::reset_gfn2_es2_device_error_cuda(device.error.get(),
                                                                device.stream) == cudaSuccess;
}

bool update_cache(UploadedES2& device) {
  return gpuxtb::detail::cuda::update_gfn2_es2_geometry_cache_cuda(
             device.batch, device.positions.get(), device.cache, device.workspace,
             device.error.get(), device.stream) == cudaSuccess;
}

bool evaluate_gpu_sequence(UploadedES2& device, std::uint64_t generation = kGeneration) {
  device.cache.geometry_generation = generation;
  return reset_error(device) && update_cache(device) &&
         gpuxtb::detail::cuda::evaluate_gfn2_es2_potential_cuda(
             device.batch, device.cache, device.charges.get(), device.potential.get(),
             device.workspace, device.error.get(), device.stream) == cudaSuccess &&
         gpuxtb::detail::cuda::add_gfn2_es2_energy_cuda(
             device.batch, device.cache, device.charges.get(), device.energies.get(),
             device.workspace, device.error.get(), device.stream) == cudaSuccess &&
         gpuxtb::detail::cuda::add_gfn2_es2_gradient_cuda(
             device.batch, device.cache, device.positions.get(), generation, device.charges.get(),
             device.gradients.get(), device.workspace, device.error.get(),
             device.stream) == cudaSuccess;
}

int test_ragged_cpu_parity_and_empty_system() {
  HostEvaluation host;
  std::string error;
  const std::vector<std::int64_t> offsets{0, 0, 3, 5};
  const std::vector<std::int32_t> atomic_numbers{8, 1, 1, 79, 1};
  const std::vector<double> positions{0.0, 0.0, 0.0,  1.4, 0.0, 1.1, -1.4, 0.0,
                                      1.1, 3.0, -0.2, 0.4, 4.8, 0.6, -0.3};
  CHECK(make_host_evaluation(offsets, atomic_numbers, positions, std::vector<double>(8u, 0.0), host,
                             error));
  CHECK(host.plan.batch_shell_offsets()[0] == 0);
  CHECK(host.plan.batch_shell_offsets()[1] == 0);
  for (std::size_t shell = 0; shell < host.charges.size(); ++shell) {
    host.charges[shell] = (shell % 2u == 0u ? 0.17 : -0.23) * static_cast<double>(shell + 1u);
  }
  CHECK(evaluate_cpu(host, kGeneration, error));
  CHECK(host.energies[0] == 0.0);

  UploadedES2 device;
  CHECK(device.initialize(host));
  CHECK(cudaMemsetAsync(device.energies.get(), 0, host.energies.size() * sizeof(double),
                        device.stream) == cudaSuccess);
  CHECK(cudaMemsetAsync(device.gradients.get(), 0, host.gradients.size() * sizeof(double),
                        device.stream) == cudaSuccess);
  CHECK(evaluate_gpu_sequence(device));
  std::uint32_t device_error = 99u;
  CHECK(device.synchronize_error(device_error));
  CHECK(device_error == static_cast<std::uint32_t>(Gfn2ES2DeviceError::kSuccess));

  std::vector<double> matrix(host.matrix.size());
  std::vector<double> potential(host.potential.size());
  std::vector<double> energies(host.energies.size());
  std::vector<double> gradients(host.gradients.size());
  CHECK(device.matrix.copy_to(matrix.data(), matrix.size()));
  CHECK(device.potential.copy_to(potential.data(), potential.size()));
  CHECK(device.energies.copy_to(energies.data(), energies.size()));
  CHECK(device.gradients.copy_to(gradients.data(), gradients.size()));
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  for (std::size_t index = 0; index < matrix.size(); ++index) {
    CHECK(near(matrix[index], host.matrix[index], 8.0e-14));
  }
  for (std::size_t index = 0; index < potential.size(); ++index) {
    CHECK(near(potential[index], host.potential[index], 3.0e-13));
  }
  for (std::size_t index = 0; index < energies.size(); ++index) {
    CHECK(near(energies[index], host.energies[index], 5.0e-13));
  }
  for (std::size_t index = 0; index < gradients.size(); ++index) {
    CHECK(near(gradients[index], host.gradients[index], 8.0e-13));
  }
  return 0;
}

bool gpu_energy_at(UploadedES2& device, const std::vector<double>& positions,
                   std::uint64_t generation, std::vector<double>& energies) {
  if (!device.positions.copy_from(positions.data(), positions.size(), device.stream) ||
      cudaMemsetAsync(device.energies.get(), 0, energies.size() * sizeof(double), device.stream) !=
          cudaSuccess) {
    return false;
  }
  device.cache.geometry_generation = generation;
  if (!reset_error(device) || !update_cache(device) ||
      gpuxtb::detail::cuda::add_gfn2_es2_energy_cuda(
          device.batch, device.cache, device.charges.get(), device.energies.get(), device.workspace,
          device.error.get(), device.stream) != cudaSuccess ||
      !device.energies.copy_to(energies.data(), energies.size(), device.stream)) {
    return false;
  }
  std::uint32_t error = 1u;
  return device.synchronize_error(error) && error == 0u;
}

int test_finite_difference_and_generation() {
  HostEvaluation host;
  std::string error;
  CHECK(make_host_evaluation({0, 3}, {8, 1, 1}, {0.0, 0.0, 0.0, 1.4, 0.0, 1.1, -1.4, 0.0, 1.1},
                             {0.2, -0.3, 0.15, -0.05}, host, error));
  CHECK(evaluate_cpu(host, kGeneration, error));
  UploadedES2 device;
  CHECK(device.initialize(host));
  CHECK(cudaMemsetAsync(device.gradients.get(), 0, host.gradients.size() * sizeof(double),
                        device.stream) == cudaSuccess);
  CHECK(reset_error(device));
  CHECK(update_cache(device));
  CHECK(gpuxtb::detail::cuda::add_gfn2_es2_gradient_cuda(
            device.batch, device.cache, device.positions.get(), kGeneration, device.charges.get(),
            device.gradients.get(), device.workspace, device.error.get(),
            device.stream) == cudaSuccess);
  std::vector<double> gpu_gradient(host.gradients.size());
  CHECK(device.gradients.copy_to(gpu_gradient.data(), gpu_gradient.size(), device.stream));
  std::uint32_t device_error = 1u;
  CHECK(device.synchronize_error(device_error));
  CHECK(device_error == 0u);

  std::vector<double> plus = host.positions;
  std::vector<double> minus = host.positions;
  constexpr double step = 2.0e-5;
  plus[3] += step;
  minus[3] -= step;
  std::vector<double> plus_energy(1u);
  std::vector<double> minus_energy(1u);
  CHECK(gpu_energy_at(device, plus, kGeneration + 1u, plus_energy));
  CHECK(gpu_energy_at(device, minus, kGeneration + 2u, minus_energy));
  const double finite_difference = (plus_energy[0] - minus_energy[0]) / (2.0 * step);
  CHECK(near(finite_difference, gpu_gradient[3], 2.0e-8));

  std::vector<double> sentinel(host.gradients.size(), 12.0);
  CHECK(device.gradients.copy_from(sentinel.data(), sentinel.size(), device.stream));
  CHECK(gpuxtb::detail::cuda::add_gfn2_es2_gradient_cuda(
            device.batch, device.cache, device.positions.get(), kGeneration + 1u,
            device.charges.get(), device.gradients.get(), device.workspace, device.error.get(),
            device.stream) == cudaErrorInvalidValue);
  std::vector<double> unchanged(sentinel.size());
  CHECK(device.gradients.copy_to(unchanged.data(), unchanged.size(), device.stream));
  CHECK(cudaStreamSynchronize(device.stream) == cudaSuccess);
  CHECK(unchanged == sentinel);

  Gfn2ES2DeviceCache wrong_cache = device.cache;
  wrong_cache.plan_token += 1u;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_es2_potential_cuda(
            device.batch, wrong_cache, device.charges.get(), device.potential.get(),
            device.workspace, device.error.get(), device.stream) == cudaErrorInvalidValue);
  return 0;
}

int test_late_failure_atomicity_sticky_and_aliases() {
  HostEvaluation host;
  std::string error;
  CHECK(make_host_evaluation({0, 2, 4}, {1, 1, 8, 1},
                             {0.0, 0.0, 0.0, 1.3, 0.0, 0.0, 3.0, 0.0, 0.0, 4.2, 0.4, -0.2},
                             {0.2, -0.1, 0.3, -0.25, 0.15}, host, error));
  CHECK(evaluate_cpu(host, kGeneration, error));
  UploadedES2 device;
  CHECK(device.initialize(host));
  CHECK(reset_error(device));
  CHECK(update_cache(device));
  std::uint32_t device_error = 1u;
  CHECK(device.synchronize_error(device_error));
  CHECK(device_error == 0u);

  std::vector<double> potential_sentinel(host.potential.size(), 71.0);
  std::vector<double> energy_sentinel(host.energies.size(), 72.0);
  std::vector<double> bad_charges = host.charges;
  bad_charges.back() = std::numeric_limits<double>::infinity();
  CHECK(device.potential.copy_from(potential_sentinel.data(), potential_sentinel.size(),
                                   device.stream));
  CHECK(device.energies.copy_from(energy_sentinel.data(), energy_sentinel.size(), device.stream));
  CHECK(device.charges.copy_from(bad_charges.data(), bad_charges.size(), device.stream));
  CHECK(reset_error(device));
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_es2_potential_cuda(
            device.batch, device.cache, device.charges.get(), device.potential.get(),
            device.workspace, device.error.get(), device.stream) == cudaSuccess);
  /* Sticky failure makes this dependent stage a no-op without a host round trip. */
  CHECK(gpuxtb::detail::cuda::add_gfn2_es2_energy_cuda(
            device.batch, device.cache, device.charges.get(), device.energies.get(),
            device.workspace, device.error.get(), device.stream) == cudaSuccess);
  std::vector<double> potential_result(potential_sentinel.size());
  std::vector<double> energy_result(energy_sentinel.size());
  CHECK(device.potential.copy_to(potential_result.data(), potential_result.size(), device.stream));
  CHECK(device.energies.copy_to(energy_result.data(), energy_result.size(), device.stream));
  CHECK(device.synchronize_error(device_error));
  CHECK(device_error == static_cast<std::uint32_t>(Gfn2ES2DeviceError::kNonfiniteShellCharge));
  CHECK(potential_result == potential_sentinel);
  CHECK(energy_result == energy_sentinel);
  CHECK(device.charges.copy_from(host.charges.data(), host.charges.size(), device.stream));

  /* A reset makes the same sequence usable again after a sticky device error. */
  CHECK(reset_error(device));
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_es2_potential_cuda(
            device.batch, device.cache, device.charges.get(), device.potential.get(),
            device.workspace, device.error.get(), device.stream) == cudaSuccess);
  CHECK(device.synchronize_error(device_error));
  CHECK(device_error == 0u);

  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_es2_potential_cuda(
            device.batch, device.cache, device.charges.get(), device.charges.get(),
            device.workspace, device.error.get(), device.stream) == cudaErrorInvalidValue);
  Gfn2ES2DeviceWorkspace partial_alias = device.workspace;
  partial_alias.shell_scratch = device.potential.get() + 1;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_es2_potential_cuda(
            device.batch, device.cache, device.charges.get(), device.potential.get(), partial_alias,
            device.error.get(), device.stream) == cudaErrorInvalidValue);
  auto* misaligned =
      reinterpret_cast<double*>(reinterpret_cast<unsigned char*>(device.potential.get()) + 1u);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_es2_potential_cuda(
            device.batch, device.cache, device.charges.get(), misaligned, device.workspace,
            device.error.get(), device.stream) == cudaErrorInvalidValue);

  Gfn2ES2DeviceBatch* managed_batch = nullptr;
  Gfn2ES2DeviceCache* managed_cache = nullptr;
  Gfn2ES2DeviceWorkspace* managed_workspace = nullptr;
  CHECK(cudaMallocManaged(reinterpret_cast<void**>(&managed_batch), sizeof(*managed_batch)) ==
        cudaSuccess);
  CHECK(cudaMallocManaged(reinterpret_cast<void**>(&managed_cache), sizeof(*managed_cache)) ==
        cudaSuccess);
  CHECK(cudaMallocManaged(reinterpret_cast<void**>(&managed_workspace),
                          sizeof(*managed_workspace)) == cudaSuccess);
  *managed_batch = device.batch;
  *managed_cache = device.cache;
  *managed_workspace = device.workspace;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_es2_potential_cuda(
            *managed_batch, device.cache, device.charges.get(),
            reinterpret_cast<double*>(managed_batch), device.workspace, device.error.get(),
            device.stream) == cudaErrorInvalidValue);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_es2_potential_cuda(
            device.batch, *managed_cache, device.charges.get(),
            reinterpret_cast<double*>(managed_cache), device.workspace, device.error.get(),
            device.stream) == cudaErrorInvalidValue);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_es2_potential_cuda(
            device.batch, device.cache, device.charges.get(),
            reinterpret_cast<double*>(managed_workspace), *managed_workspace, device.error.get(),
            device.stream) == cudaErrorInvalidValue);
  CHECK(cudaFree(managed_workspace) == cudaSuccess);
  CHECK(cudaFree(managed_cache) == cudaSuccess);
  CHECK(cudaFree(managed_batch) == cudaSuccess);

  std::vector<double> matrix_sentinel(host.matrix.size(), 81.0);
  std::vector<double> bad_positions = host.positions;
  bad_positions.back() = std::numeric_limits<double>::infinity();
  CHECK(device.matrix.copy_from(matrix_sentinel.data(), matrix_sentinel.size(), device.stream));
  CHECK(device.positions.copy_from(bad_positions.data(), bad_positions.size(), device.stream));
  CHECK(reset_error(device));
  CHECK(update_cache(device));
  std::vector<double> matrix_result(matrix_sentinel.size());
  CHECK(device.matrix.copy_to(matrix_result.data(), matrix_result.size(), device.stream));
  CHECK(device.synchronize_error(device_error));
  CHECK(device_error == static_cast<std::uint32_t>(Gfn2ES2DeviceError::kNonfinitePosition));
  CHECK(matrix_result == matrix_sentinel);

  /* A malformed device-side ragged partition must fail before publication. */
  CHECK(device.positions.copy_from(host.positions.data(), host.positions.size(), device.stream));
  std::vector<std::int64_t> bad_shell_offsets = host.plan.batch_shell_offsets();
  bad_shell_offsets[1] = host.plan.total_shells() + 1;
  CHECK(device.batch_shell_offsets.copy_from(bad_shell_offsets.data(), bad_shell_offsets.size(),
                                             device.stream));
  CHECK(device.matrix.copy_from(matrix_sentinel.data(), matrix_sentinel.size(), device.stream));
  CHECK(reset_error(device));
  CHECK(update_cache(device));
  CHECK(device.matrix.copy_to(matrix_result.data(), matrix_result.size(), device.stream));
  CHECK(device.synchronize_error(device_error));
  CHECK(device_error == static_cast<std::uint32_t>(Gfn2ES2DeviceError::kInvalidOffsets));
  CHECK(matrix_result == matrix_sentinel);

  /* Host descriptor count mismatches are rejected without enqueueing kernels. */
  Gfn2ES2DeviceBatch wrong_count = device.batch;
  --wrong_count.shell_to_atom_count;
  CHECK(gpuxtb::detail::cuda::update_gfn2_es2_geometry_cache_cuda(
            wrong_count, device.positions.get(), device.cache, device.workspace, device.error.get(),
            device.stream) == cudaErrorInvalidValue);

  /* Restore the topology and prove another reset permits a successful update. */
  CHECK(device.batch_shell_offsets.copy_from(host.plan.batch_shell_offsets().data(),
                                             host.plan.batch_shell_offsets().size(),
                                             device.stream));
  CHECK(reset_error(device));
  CHECK(update_cache(device));
  CHECK(device.synchronize_error(device_error));
  CHECK(device_error == 0u);

  std::vector<double> invalid_matrix = host.matrix;
  invalid_matrix.back() = std::numeric_limits<double>::quiet_NaN();
  CHECK(device.matrix.copy_from(invalid_matrix.data(), invalid_matrix.size(), device.stream));
  CHECK(device.potential.copy_from(potential_sentinel.data(), potential_sentinel.size(),
                                   device.stream));
  CHECK(reset_error(device));
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_es2_potential_cuda(
            device.batch, device.cache, device.charges.get(), device.potential.get(),
            device.workspace, device.error.get(), device.stream) == cudaSuccess);
  CHECK(device.potential.copy_to(potential_result.data(), potential_result.size(), device.stream));
  CHECK(device.synchronize_error(device_error));
  CHECK(device_error == static_cast<std::uint32_t>(Gfn2ES2DeviceError::kInvalidCacheMatrix));
  CHECK(potential_result == potential_sentinel);
  CHECK(reset_error(device));
  CHECK(update_cache(device));
  CHECK(device.synchronize_error(device_error));
  CHECK(device_error == 0u);

  const double nonfinite_seed = std::numeric_limits<double>::infinity();
  CHECK(device.energies.copy_from(&nonfinite_seed, 1u, device.stream));
  CHECK(device.potential.copy_from(potential_sentinel.data(), potential_sentinel.size(),
                                   device.stream));
  CHECK(reset_error(device));
  CHECK(gpuxtb::detail::cuda::add_gfn2_es2_energy_cuda(
            device.batch, device.cache, device.charges.get(), device.energies.get(),
            device.workspace, device.error.get(), device.stream) == cudaSuccess);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_es2_potential_cuda(
            device.batch, device.cache, device.charges.get(), device.potential.get(),
            device.workspace, device.error.get(), device.stream) == cudaSuccess);
  CHECK(device.potential.copy_to(potential_result.data(), potential_result.size(), device.stream));
  CHECK(device.synchronize_error(device_error));
  CHECK(device_error == static_cast<std::uint32_t>(Gfn2ES2DeviceError::kNonfiniteEnergySeed));
  CHECK(potential_result == potential_sentinel);

  /* A finite late-batch overflow must preserve every energy seed. */
  std::vector<double> overflow_charges = host.charges;
  overflow_charges.back() = std::numeric_limits<double>::max();
  CHECK(device.charges.copy_from(overflow_charges.data(), overflow_charges.size(), device.stream));
  CHECK(device.energies.copy_from(energy_sentinel.data(), energy_sentinel.size(), device.stream));
  CHECK(reset_error(device));
  CHECK(gpuxtb::detail::cuda::add_gfn2_es2_energy_cuda(
            device.batch, device.cache, device.charges.get(), device.energies.get(),
            device.workspace, device.error.get(), device.stream) == cudaSuccess);
  CHECK(device.energies.copy_to(energy_result.data(), energy_result.size(), device.stream));
  CHECK(device.synchronize_error(device_error));
  CHECK(device_error == static_cast<std::uint32_t>(Gfn2ES2DeviceError::kNonfiniteEnergyArithmetic));
  CHECK(energy_result == energy_sentinel);
  return 0;
}

int test_gradient_seed_overflow_atomicity() {
  HostEvaluation host;
  std::string error;
  CHECK(make_host_evaluation({0, 2}, {1, 1}, {0.0, 0.0, 0.0, 1.0, 0.0, 0.0}, {1.0e154, 1.0e154},
                             host, error));
  UploadedES2 device;
  CHECK(device.initialize(host));
  CHECK(reset_error(device));
  CHECK(update_cache(device));
  std::uint32_t device_error = 1u;
  CHECK(device.synchronize_error(device_error));
  CHECK(device_error == 0u);

  std::vector<double> gradient_seed(host.gradients.size(), 0.0);
  gradient_seed[0] = std::numeric_limits<double>::max();
  CHECK(device.gradients.copy_from(gradient_seed.data(), gradient_seed.size(), device.stream));
  CHECK(reset_error(device));
  CHECK(gpuxtb::detail::cuda::add_gfn2_es2_gradient_cuda(
            device.batch, device.cache, device.positions.get(), kGeneration, device.charges.get(),
            device.gradients.get(), device.workspace, device.error.get(),
            device.stream) == cudaSuccess);
  std::vector<double> gradient_result(gradient_seed.size());
  CHECK(device.gradients.copy_to(gradient_result.data(), gradient_result.size(), device.stream));
  CHECK(device.synchronize_error(device_error));
  CHECK(device_error ==
        static_cast<std::uint32_t>(Gfn2ES2DeviceError::kNonfiniteGradientArithmetic));
  CHECK(gradient_result == gradient_seed);

  /* A nonfinite seed is diagnosed separately and must not publish any coordinate. */
  std::vector<double> nonfinite_seed(host.gradients.size(), 34.0);
  nonfinite_seed[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK(device.gradients.copy_from(nonfinite_seed.data(), nonfinite_seed.size(), device.stream));
  CHECK(reset_error(device));
  CHECK(gpuxtb::detail::cuda::add_gfn2_es2_gradient_cuda(
            device.batch, device.cache, device.positions.get(), kGeneration, device.charges.get(),
            device.gradients.get(), device.workspace, device.error.get(),
            device.stream) == cudaSuccess);
  CHECK(device.gradients.copy_to(gradient_result.data(), gradient_result.size(), device.stream));
  CHECK(device.synchronize_error(device_error));
  CHECK(device_error == static_cast<std::uint32_t>(Gfn2ES2DeviceError::kNonfiniteGradientSeed));
  CHECK(std::isnan(gradient_result[0]));
  for (std::size_t coordinate = 1; coordinate < gradient_result.size(); ++coordinate) {
    CHECK(gradient_result[coordinate] == nonfinite_seed[coordinate]);
  }

  /* Charge validation also covers a one-atom system with no atom-pair loop. */
  HostEvaluation single_atom;
  CHECK(make_host_evaluation({0, 1}, {1}, {0.0, 0.0, 0.0},
                             {std::numeric_limits<double>::infinity()}, single_atom, error));
  UploadedES2 single_device;
  CHECK(single_device.initialize(single_atom));
  CHECK(reset_error(single_device));
  CHECK(update_cache(single_device));
  std::vector<double> single_sentinel(single_atom.gradients.size(), 35.0);
  CHECK(single_device.gradients.copy_from(single_sentinel.data(), single_sentinel.size(),
                                          single_device.stream));
  CHECK(gpuxtb::detail::cuda::add_gfn2_es2_gradient_cuda(
            single_device.batch, single_device.cache, single_device.positions.get(), kGeneration,
            single_device.charges.get(), single_device.gradients.get(), single_device.workspace,
            single_device.error.get(), single_device.stream) == cudaSuccess);
  std::vector<double> single_result(single_sentinel.size());
  CHECK(single_device.gradients.copy_to(single_result.data(), single_result.size(),
                                        single_device.stream));
  CHECK(single_device.synchronize_error(device_error));
  CHECK(device_error == static_cast<std::uint32_t>(Gfn2ES2DeviceError::kNonfiniteShellCharge));
  CHECK(single_result == single_sentinel);
  return 0;
}

int test_energy_seed_is_added_after_batch_reduction() {
  HostEvaluation host;
  std::string error;
  CHECK(make_host_evaluation({0, 2}, {1, 1}, {0.0, 0.0, 0.0, 1.0, 0.0, 0.0}, {1.0e153, -2.0e153},
                             host, error));
  CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
            host.plan, host.positions.data(), kGeneration, host.matrix.data(), host.matrix.size(),
            host.workspace, host.cache, error) == GPUXTB_STATUS_SUCCESS);
  host.energies[0] = -std::numeric_limits<double>::max();
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(host.plan, host.cache, host.charges.data(),
                                                 host.energies.data(), host.workspace,
                                                 error) == GPUXTB_STATUS_SUCCESS);
  CHECK(std::isfinite(host.energies[0]));
  CHECK(host.energies[0] > -std::numeric_limits<double>::max());

  UploadedES2 device;
  CHECK(device.initialize(host));
  const double seed = -std::numeric_limits<double>::max();
  CHECK(device.energies.copy_from(&seed, 1u, device.stream));
  CHECK(reset_error(device));
  CHECK(update_cache(device));
  CHECK(gpuxtb::detail::cuda::add_gfn2_es2_energy_cuda(
            device.batch, device.cache, device.charges.get(), device.energies.get(),
            device.workspace, device.error.get(), device.stream) == cudaSuccess);
  double result = 0.0;
  CHECK(device.energies.copy_to(&result, 1u, device.stream));
  std::uint32_t device_error = 1u;
  CHECK(device.synchronize_error(device_error));
  CHECK(device_error == 0u);
  CHECK(std::isfinite(result));
  CHECK(result > seed);
  CHECK(near(result, host.energies[0]));
  return 0;
}

int test_extreme_arithmetic_and_large_stride() {
  HostEvaluation host;
  std::string error;
  CHECK(make_host_evaluation({0, 1}, {8}, {0.0, 0.0, 0.0}, {0.1, -0.2}, host, error));
  UploadedES2 device;
  CHECK(device.initialize(host));
  std::vector<double> maximum_hardness(host.plan.total_shells(),
                                       std::numeric_limits<double>::max());
  CHECK(device.shell_hardness.copy_from(maximum_hardness.data(), maximum_hardness.size(),
                                        device.stream));
  CHECK(reset_error(device));
  CHECK(update_cache(device));
  std::uint32_t device_error = 1u;
  CHECK(device.synchronize_error(device_error));
  CHECK(device_error == 0u);
  std::vector<double> matrix(host.matrix.size());
  CHECK(device.matrix.copy_to(matrix.data(), matrix.size(), device.stream));
  CHECK(cudaStreamSynchronize(device.stream) == cudaSuccess);
  CHECK(matrix[0] == std::numeric_limits<double>::max());

  /* Preserve CPU addition order for finite subnormal arithmetic-hardness sums. */
  constexpr double first_hardness = 0x0.9e7d84a32ae6dp-1022;
  constexpr double second_hardness = 0x0.837fba39941a9p-1022;
  constexpr double cpu_average = 0x0.90fe9f6e5f80bp-1022;
  const std::vector<double> finite_subnormal_hardness{first_hardness, second_hardness};
  CHECK(finite_subnormal_hardness.size() == static_cast<std::size_t>(host.plan.total_shells()));
  CHECK(device.shell_hardness.copy_from(finite_subnormal_hardness.data(),
                                        finite_subnormal_hardness.size(), device.stream));
  CHECK(reset_error(device));
  CHECK(update_cache(device));
  CHECK(device.matrix.copy_to(matrix.data(), matrix.size(), device.stream));
  CHECK(device.synchronize_error(device_error));
  CHECK(device_error == 0u);
  CHECK(matrix[1] == cpu_average);
  CHECK(matrix[2] == cpu_average);

  std::vector<double> subnormal_hardness(host.plan.total_shells(),
                                         std::numeric_limits<double>::denorm_min());
  std::vector<double> sentinel(host.matrix.size(), 91.0);
  CHECK(device.shell_hardness.copy_from(subnormal_hardness.data(), subnormal_hardness.size(),
                                        device.stream));
  CHECK(device.matrix.copy_from(sentinel.data(), sentinel.size(), device.stream));
  CHECK(reset_error(device));
  CHECK(update_cache(device));
  CHECK(device.matrix.copy_to(matrix.data(), matrix.size(), device.stream));
  CHECK(device.synchronize_error(device_error));
  CHECK(device_error ==
        static_cast<std::uint32_t>(Gfn2ES2DeviceError::kNonfiniteHardnessArithmetic));
  CHECK(matrix == sentinel);

  HostEvaluation separated;
  const double maximum = std::numeric_limits<double>::max();
  CHECK(make_host_evaluation({0, 2}, {1, 1}, {-maximum, 0.0, 0.0, maximum, 0.0, 0.0}, {0.1, -0.2},
                             separated, error));
  UploadedES2 separated_device;
  CHECK(separated_device.initialize(separated));
  std::vector<double> separated_sentinel(separated.matrix.size(), 92.0);
  CHECK(separated_device.matrix.copy_from(separated_sentinel.data(), separated_sentinel.size(),
                                          separated_device.stream));
  CHECK(reset_error(separated_device));
  CHECK(update_cache(separated_device));
  std::vector<double> separated_matrix(separated.matrix.size());
  CHECK(separated_device.matrix.copy_to(separated_matrix.data(), separated_matrix.size(),
                                        separated_device.stream));
  CHECK(separated_device.synchronize_error(device_error));
  CHECK(device_error ==
        static_cast<std::uint32_t>(Gfn2ES2DeviceError::kCoordinateDifferenceOverflow));
  CHECK(separated_matrix == separated_sentinel);

  constexpr std::size_t atom_count = 300u;
  std::vector<std::int64_t> offsets{0, static_cast<std::int64_t>(atom_count)};
  std::vector<std::int32_t> atomic_numbers(atom_count, 1);
  std::vector<double> positions(atom_count * 3u, 0.0);
  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    positions[atom * 3u] = 1.1 * static_cast<double>(atom);
  }
  std::vector<double> charges(atom_count);
  for (std::size_t shell = 0; shell < charges.size(); ++shell) {
    charges[shell] = (shell % 2u == 0u ? 0.001 : -0.0015);
  }
  HostEvaluation large;
  CHECK(make_host_evaluation(offsets, atomic_numbers, positions, charges, large, error));
  CHECK(evaluate_cpu(large, kGeneration, error));
  UploadedES2 large_device;
  CHECK(large_device.initialize(large));
  CHECK(reset_error(large_device));
  CHECK(update_cache(large_device));
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_es2_potential_cuda(
            large_device.batch, large_device.cache, large_device.charges.get(),
            large_device.potential.get(), large_device.workspace, large_device.error.get(),
            large_device.stream) == cudaSuccess);
  std::vector<double> large_potential(large.potential.size());
  CHECK(large_device.potential.copy_to(large_potential.data(), large_potential.size(),
                                       large_device.stream));
  CHECK(large_device.synchronize_error(device_error));
  CHECK(device_error == 0u);
  for (std::size_t shell = 0; shell < large_potential.size(); ++shell) {
    CHECK(near(large_potential[shell], large.potential[shell], 1.0e-12));
  }

  CHECK(cudaMemsetAsync(large_device.energies.get(), 0, large.energies.size() * sizeof(double),
                        large_device.stream) == cudaSuccess);
  CHECK(cudaMemsetAsync(large_device.gradients.get(), 0, large.gradients.size() * sizeof(double),
                        large_device.stream) == cudaSuccess);
  CHECK(reset_error(large_device));
  CHECK(gpuxtb::detail::cuda::add_gfn2_es2_energy_cuda(
            large_device.batch, large_device.cache, large_device.charges.get(),
            large_device.energies.get(), large_device.workspace, large_device.error.get(),
            large_device.stream) == cudaSuccess);
  CHECK(gpuxtb::detail::cuda::add_gfn2_es2_gradient_cuda(
            large_device.batch, large_device.cache, large_device.positions.get(), kGeneration,
            large_device.charges.get(), large_device.gradients.get(), large_device.workspace,
            large_device.error.get(), large_device.stream) == cudaSuccess);
  std::vector<double> large_energy(large.energies.size());
  std::vector<double> large_gradient(large.gradients.size());
  CHECK(
      large_device.energies.copy_to(large_energy.data(), large_energy.size(), large_device.stream));
  CHECK(large_device.gradients.copy_to(large_gradient.data(), large_gradient.size(),
                                       large_device.stream));
  CHECK(large_device.synchronize_error(device_error));
  CHECK(device_error == 0u);
  for (std::size_t system = 0; system < large_energy.size(); ++system) {
    CHECK(near(large_energy[system], large.energies[system], 1.0e-12));
  }
  for (std::size_t coordinate = 0; coordinate < large_gradient.size(); ++coordinate) {
    CHECK(near(large_gradient[coordinate], large.gradients[coordinate], 1.0e-11));
  }
  return 0;
}

int test_cuda_graph_capture_and_replay() {
  HostEvaluation host;
  std::string error;
  CHECK(make_host_evaluation({0, 3}, {8, 1, 1}, {0.0, 0.0, 0.0, 1.4, 0.0, 1.1, -1.4, 0.0, 1.1},
                             {0.2, -0.3, 0.15, -0.05}, host, error));
  CHECK(evaluate_cpu(host, kGeneration, error));
  UploadedES2 device;
  CHECK(device.initialize(host));
  device.cache.geometry_generation = kGeneration;

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CHECK(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeGlobal) == cudaSuccess);
  CHECK(reset_error(device));
  CHECK(cudaMemsetAsync(device.energies.get(), 0, host.energies.size() * sizeof(double),
                        device.stream) == cudaSuccess);
  CHECK(cudaMemsetAsync(device.gradients.get(), 0, host.gradients.size() * sizeof(double),
                        device.stream) == cudaSuccess);
  CHECK(update_cache(device));
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_es2_potential_cuda(
            device.batch, device.cache, device.charges.get(), device.potential.get(),
            device.workspace, device.error.get(), device.stream) == cudaSuccess);
  CHECK(gpuxtb::detail::cuda::add_gfn2_es2_energy_cuda(
            device.batch, device.cache, device.charges.get(), device.energies.get(),
            device.workspace, device.error.get(), device.stream) == cudaSuccess);
  CHECK(gpuxtb::detail::cuda::add_gfn2_es2_gradient_cuda(
            device.batch, device.cache, device.positions.get(), kGeneration, device.charges.get(),
            device.gradients.get(), device.workspace, device.error.get(),
            device.stream) == cudaSuccess);
  CHECK(cudaStreamEndCapture(device.stream, &graph) == cudaSuccess);
  CHECK(graph != nullptr);
  CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0) == cudaSuccess);
  CHECK(cudaGraphLaunch(executable, device.stream) == cudaSuccess);
  CHECK(cudaGraphLaunch(executable, device.stream) == cudaSuccess);
  std::uint32_t device_error = 1u;
  CHECK(device.synchronize_error(device_error));
  CHECK(device_error == 0u);
  std::vector<double> energy(host.energies.size());
  std::vector<double> gradient(host.gradients.size());
  CHECK(device.energies.copy_to(energy.data(), energy.size()));
  CHECK(device.gradients.copy_to(gradient.data(), gradient.size()));
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  for (std::size_t index = 0; index < energy.size(); ++index) {
    CHECK(near(energy[index], host.energies[index], 5.0e-13));
  }
  for (std::size_t index = 0; index < gradient.size(); ++index) {
    CHECK(near(gradient[index], host.gradients[index], 8.0e-13));
  }
  CHECK(cudaGraphExecDestroy(executable) == cudaSuccess);
  CHECK(cudaGraphDestroy(graph) == cudaSuccess);
  return 0;
}

}  // namespace

int main() {
  if (const int line = test_ragged_cpu_parity_and_empty_system(); line != 0) {
    return line;
  }
  if (const int line = test_finite_difference_and_generation(); line != 0) {
    return line;
  }
  if (const int line = test_late_failure_atomicity_sticky_and_aliases(); line != 0) {
    return line;
  }
  if (const int line = test_gradient_seed_overflow_atomicity(); line != 0) {
    return line;
  }
  if (const int line = test_energy_seed_is_added_after_batch_reduction(); line != 0) {
    return line;
  }
  if (const int line = test_extreme_arithmetic_and_large_stride(); line != 0) {
    return line;
  }
  if (const int line = test_cuda_graph_capture_and_replay(); line != 0) {
    return line;
  }
  return 0;
}
