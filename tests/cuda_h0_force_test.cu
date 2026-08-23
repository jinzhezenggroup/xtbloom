#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_density.cuh"
#include "backends/cuda/gfn2_h0_force.cuh"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/h0.hpp"
#include "model/gfn2/integrals.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using xtbloom::detail::cuda::Gfn2ForceDeviceActivity;
using xtbloom::detail::cuda::Gfn2H0DevicePlan;
using xtbloom::detail::cuda::Gfn2H0ForceDeviceError;
using xtbloom::detail::cuda::Gfn2H0ForceDeviceInput;
using xtbloom::detail::cuda::Gfn2H0ForceDeviceOutput;
using xtbloom::detail::cuda::Gfn2H0ForceDeviceWorkspace;
using xtbloom::detail::cuda::Gfn2IntegralDeviceBatch;
using xtbloom::detail::cuda::kGfn2IntegralLinearBlockBudget;
using xtbloom::detail::cuda::select_gfn2_density_contraction_tiles;
using xtbloom::detail::gfn2::BasisPlan;
using xtbloom::detail::gfn2::H0Plan;
using xtbloom::detail::gfn2::IntegralPlan;

constexpr std::uint64_t kPlanToken = 0x64b4ec22891d5307ULL;

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
    count_ = std::max<std::size_t>(count, 1u);
    return cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T));
  }

  cudaError_t copy_from(const T* source, std::size_t count, cudaStream_t stream = nullptr) {
    if (source == nullptr || count > count_) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(data_, source, count * sizeof(T), cudaMemcpyHostToDevice, stream);
  }

  cudaError_t copy_to(T* target, std::size_t count, cudaStream_t stream = nullptr) const {
    if (target == nullptr || count > count_) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(target, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
  }

  T* get() { return data_; }
  const T* get() const { return data_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

template <typename T>
cudaError_t upload(DeviceBuffer<T>& target, const std::vector<T>& source, cudaStream_t stream) {
  cudaError_t status = target.allocate(source.size());
  return status == cudaSuccess ? target.copy_from(source.data(), source.size(), stream) : status;
}

bool near(double actual, double expected, double absolute = 3.0e-12, double relative = 3.0e-12) {
  return std::abs(actual - expected) <=
         absolute + relative * std::max(std::abs(actual), std::abs(expected));
}

int count_kernel_grid(cudaGraph_t graph, dim3 expected_grid, std::size_t& matches) {
  std::size_t node_count = 0u;
  CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &node_count));
  std::vector<cudaGraphNode_t> nodes(node_count);
  CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &node_count));
  matches = 0u;
  for (const cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type{};
    CUDA_CHECK(cudaGraphNodeGetType(node, &type));
    if (type != cudaGraphNodeTypeKernel) {
      continue;
    }
    cudaKernelNodeParams parameters{};
    CUDA_CHECK(cudaGraphKernelNodeGetParams(node, &parameters));
    const dim3 grid = parameters.gridDim;
    if (grid.x == expected_grid.x && grid.y == expected_grid.y && grid.z == expected_grid.z) {
      ++matches;
    }
  }
  return 0;
}

struct HostCase {
  BasisPlan basis;
  IntegralPlan integrals;
  H0Plan h0;
  std::int64_t maximum_system_shells = 0;
  std::vector<double> positions;
  std::vector<double> coordination;
  std::vector<double> overlap;
  std::vector<double> density;
  std::vector<double> weighted_density;
  std::vector<double> overlap_seed;
  std::vector<double> coordination_seed;
  std::vector<double> gradient_seed;
};

bool make_case(HostCase& data, std::string& error, std::size_t large_singleton_atoms = 0u) {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  if (large_singleton_atoms == 0u) {
    /* H2, bent H2O, and LiH exercise s, p, directed shell pairs, and ragged sizes. */
    atom_offsets = {0, 2, 5, 7};
    atomic_numbers = {1, 1, 8, 1, 1, 3, 1};
    data.positions = {
        0.00,  0.00, -0.71, 0.00, 0.00, 0.71, 4.10,  -0.20, 0.13,  5.48, 0.37,
        -0.22, 3.53, 1.19,  0.61, 8.00, 0.10, -1.49, 8.31,  -0.27, 1.50,
    };
  } else {
    atom_offsets = {0, static_cast<std::int64_t>(large_singleton_atoms)};
    atomic_numbers.assign(large_singleton_atoms, 1);
    data.positions.reserve(3u * large_singleton_atoms);
    for (std::size_t atom = 0; atom < large_singleton_atoms; ++atom) {
      data.positions.push_back(2.25 * static_cast<double>(atom));
      data.positions.push_back(0.13 * static_cast<double>(atom % 3u));
      data.positions.push_back(-0.09 * static_cast<double>(atom % 5u));
    }
  }
  const std::int64_t batch_size = static_cast<std::int64_t>(atom_offsets.size() - 1u);
  if (xtbloom::detail::gfn2::make_basis_plan(
          batch_size, static_cast<std::int64_t>(atomic_numbers.size()), atom_offsets.data(),
          atomic_numbers.data(), data.basis, error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_integral_plan(data.basis, data.integrals, error) !=
          XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_h0_plan(data.basis, data.integrals, atomic_numbers.data(),
                                          data.h0, error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  for (std::int64_t system = 0; system < data.basis.batch_size; ++system) {
    const std::size_t index = static_cast<std::size_t>(system);
    data.maximum_system_shells =
        std::max(data.maximum_system_shells, data.basis.batch_shell_offsets[index + 1u] -
                                                 data.basis.batch_shell_offsets[index]);
  }
  const std::size_t atoms = static_cast<std::size_t>(data.basis.total_atoms);
  const std::size_t matrices = static_cast<std::size_t>(data.integrals.total_matrix_elements);
  data.coordination.resize(atoms);
  data.overlap.resize(matrices);
  data.density.resize(matrices);
  data.weighted_density.resize(matrices);
  data.overlap_seed.resize(matrices);
  data.coordination_seed.resize(atoms);
  data.gradient_seed.resize(3u * atoms);
  for (std::size_t atom = 0; atom < atoms; ++atom) {
    data.coordination[atom] = 0.43 + 0.09 * static_cast<double>((atom * 7u) % 13u);
    data.coordination_seed[atom] = -0.017 + 0.004 * static_cast<double>(atom);
    for (std::size_t axis = 0; axis < 3u; ++axis) {
      data.gradient_seed[3u * atom + axis] =
          0.003 * static_cast<double>((atom + 2u * axis) % 7u) - 0.008;
    }
  }
  std::vector<double> workspace((data.integrals.workspace_size_bytes + sizeof(double) - 1u) /
                                sizeof(double));
  if (xtbloom::detail::gfn2::evaluate_overlap_cpu(
          data.basis, data.integrals, data.positions.data(), data.overlap.data(), workspace.data(),
          workspace.size() * sizeof(double), error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  for (std::int64_t system = 0; system < data.basis.batch_size; ++system) {
    const std::size_t system_index = static_cast<std::size_t>(system);
    const std::int64_t begin = data.integrals.matrix_offsets[system_index];
    const std::int64_t end = data.integrals.matrix_offsets[system_index + 1u];
    const std::int64_t orbitals = data.basis.batch_orbital_offsets[system_index + 1u] -
                                  data.basis.batch_orbital_offsets[system_index];
    for (std::int64_t row = 0; row < orbitals; ++row) {
      for (std::int64_t column = 0; column <= row; ++column) {
        const double density =
            0.11 + 0.007 * static_cast<double>((row * 13 + column * 5 + system * 3) % 29);
        const double weighted =
            -0.19 + 0.006 * static_cast<double>((row * 7 + column * 17 + system) % 31);
        data.density[static_cast<std::size_t>(begin + row * orbitals + column)] = density;
        data.density[static_cast<std::size_t>(begin + column * orbitals + row)] = density;
        data.weighted_density[static_cast<std::size_t>(begin + row * orbitals + column)] = weighted;
        data.weighted_density[static_cast<std::size_t>(begin + column * orbitals + row)] = weighted;
      }
    }
    for (std::int64_t matrix = begin; matrix < end; ++matrix) {
      data.overlap_seed[static_cast<std::size_t>(matrix)] =
          0.021 - 0.002 * static_cast<double>((matrix * 11) % 17);
    }
  }
  return true;
}

struct Expected {
  std::vector<double> overlap_adjoint;
  std::vector<double> coordination_adjoint;
  std::vector<double> gradients;
};

bool cpu_expected(const HostCase& data, Expected& expected, std::string& error) {
  expected.overlap_adjoint = data.overlap_seed;
  expected.coordination_adjoint = data.coordination_seed;
  expected.gradients = data.gradient_seed;
  if (xtbloom::detail::gfn2::add_h0_vjp_cpu(
          data.basis, data.integrals, data.h0, data.positions.data(), data.coordination.data(),
          data.overlap.data(), data.density.data(), expected.overlap_adjoint.data(),
          expected.coordination_adjoint.data(), expected.gradients.data(),
          error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  for (std::size_t matrix = 0; matrix < expected.overlap_adjoint.size(); ++matrix) {
    expected.overlap_adjoint[matrix] -= data.weighted_density[matrix];
  }
  return true;
}

double stationary_core_energy(const HostCase& data, const std::vector<double>& positions,
                              const std::vector<double>& coordination,
                              const std::vector<double>& overlap, std::string& error) {
  std::vector<double> hamiltonian(overlap.size());
  if (xtbloom::detail::gfn2::evaluate_h0_cpu(data.basis, data.integrals, data.h0, positions.data(),
                                             coordination.data(), overlap.data(),
                                             hamiltonian.data(), error) != XTBLOOM_STATUS_SUCCESS) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::inner_product(data.density.begin(), data.density.end(), hamiltonian.begin(), 0.0) -
         std::inner_product(data.weighted_density.begin(), data.weighted_density.end(),
                            overlap.begin(), 0.0);
}

struct DeviceFixture {
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> batch_shell_offsets;
  DeviceBuffer<std::int64_t> batch_orbital_offsets;
  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<std::int64_t> shell_pair_offsets;
  DeviceBuffer<std::int64_t> atom_shell_offsets;
  DeviceBuffer<std::int64_t> shell_orbital_offsets;
  DeviceBuffer<std::int64_t> shell_to_atom;
  DeviceBuffer<double> atomic_radii;
  DeviceBuffer<double> shell_levels;
  DeviceBuffer<double> shell_coordination_scale;
  DeviceBuffer<double> shell_polynomial;
  DeviceBuffer<double> shell_pair_scale;
  DeviceBuffer<double> positions;
  DeviceBuffer<double> coordination;
  DeviceBuffer<double> overlap;
  DeviceBuffer<double> density;
  DeviceBuffer<double> weighted_density;
  DeviceBuffer<std::uint8_t> requested;
  DeviceBuffer<xtbloom_status_t> statuses;
  DeviceBuffer<double> overlap_adjoint;
  DeviceBuffer<double> coordination_adjoint;
  DeviceBuffer<double> gradients;
  DeviceBuffer<double> overlap_scratch;
  DeviceBuffer<double> coordination_scratch;
  DeviceBuffer<double> gradient_scratch;
  DeviceBuffer<std::uint32_t> sequence_active;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;

  cudaError_t initialize(const HostCase& host, cudaStream_t stream) {
    cudaError_t status = upload(atom_offsets, host.basis.atom_offsets, stream);
#define UPLOAD(field, source)               \
  if (status == cudaSuccess) {              \
    status = upload(field, source, stream); \
  }
    UPLOAD(batch_shell_offsets, host.basis.batch_shell_offsets)
    UPLOAD(batch_orbital_offsets, host.basis.batch_orbital_offsets)
    UPLOAD(matrix_offsets, host.integrals.matrix_offsets)
    UPLOAD(shell_pair_offsets, host.h0.shell_pair_offsets)
    UPLOAD(atom_shell_offsets, host.basis.atom_shell_offsets)
    UPLOAD(shell_orbital_offsets, host.basis.shell_orbital_offsets)
    UPLOAD(shell_to_atom, host.basis.shell_to_atom)
    UPLOAD(atomic_radii, host.h0.atomic_radii)
    UPLOAD(shell_levels, host.h0.shell_levels)
    UPLOAD(shell_coordination_scale, host.h0.shell_coordination_scale)
    UPLOAD(shell_polynomial, host.h0.shell_polynomial)
    UPLOAD(shell_pair_scale, host.h0.shell_pair_scale)
    UPLOAD(positions, host.positions)
    UPLOAD(coordination, host.coordination)
    UPLOAD(overlap, host.overlap)
    UPLOAD(density, host.density)
    UPLOAD(weighted_density, host.weighted_density)
#undef UPLOAD
    const std::size_t systems = static_cast<std::size_t>(host.basis.batch_size);
    const std::size_t atoms = static_cast<std::size_t>(host.basis.total_atoms);
    const std::size_t matrices = static_cast<std::size_t>(host.integrals.total_matrix_elements);
#define ALLOCATE(field, count)        \
  if (status == cudaSuccess) {        \
    status = field.allocate((count)); \
  }
    ALLOCATE(requested, systems)
    ALLOCATE(statuses, systems)
    ALLOCATE(overlap_adjoint, matrices)
    ALLOCATE(coordination_adjoint, atoms)
    ALLOCATE(gradients, 3u * atoms)
    ALLOCATE(overlap_scratch, matrices)
    ALLOCATE(coordination_scratch, atoms)
    ALLOCATE(gradient_scratch, 3u * atoms)
    ALLOCATE(sequence_active, 1u)
    ALLOCATE(system_errors, systems)
    ALLOCATE(device_error, 1u)
#undef ALLOCATE
    return status;
  }

  Gfn2IntegralDeviceBatch batch(const HostCase& host) const {
    Gfn2IntegralDeviceBatch value{};
    value.batch_size = host.basis.batch_size;
    value.total_atoms = host.basis.total_atoms;
    value.total_shells = host.basis.total_shells;
    value.total_orbitals = host.basis.total_orbitals;
    value.total_matrix_elements = host.integrals.total_matrix_elements;
    value.total_shell_pair_elements = host.h0.shell_pair_offsets.back();
    value.maximum_system_shells = host.maximum_system_shells;
    value.linear_tiles_per_system = 1;
    value.plan_token = kPlanToken;
    value.atom_offset_count = static_cast<std::int64_t>(host.basis.atom_offsets.size());
    value.batch_shell_offset_count =
        static_cast<std::int64_t>(host.basis.batch_shell_offsets.size());
    value.batch_orbital_offset_count =
        static_cast<std::int64_t>(host.basis.batch_orbital_offsets.size());
    value.matrix_offset_count = static_cast<std::int64_t>(host.integrals.matrix_offsets.size());
    value.shell_pair_offset_count = static_cast<std::int64_t>(host.h0.shell_pair_offsets.size());
    value.atom_shell_offset_count = static_cast<std::int64_t>(host.basis.atom_shell_offsets.size());
    value.shell_orbital_offset_count =
        static_cast<std::int64_t>(host.basis.shell_orbital_offsets.size());
    value.shell_to_atom_count = static_cast<std::int64_t>(host.basis.shell_to_atom.size());
    value.atom_offsets = atom_offsets.get();
    value.batch_shell_offsets = batch_shell_offsets.get();
    value.batch_orbital_offsets = batch_orbital_offsets.get();
    value.matrix_offsets = matrix_offsets.get();
    value.shell_pair_offsets = shell_pair_offsets.get();
    value.atom_shell_offsets = atom_shell_offsets.get();
    value.shell_orbital_offsets = shell_orbital_offsets.get();
    value.shell_to_atom = shell_to_atom.get();
    return value;
  }

  Gfn2H0DevicePlan plan(const HostCase& host) const {
    return {host.basis.total_atoms,
            host.basis.total_shells,
            host.basis.total_shells,
            host.basis.total_shells,
            host.h0.shell_pair_offsets.back(),
            kPlanToken,
            atomic_radii.get(),
            shell_levels.get(),
            shell_coordination_scale.get(),
            shell_polynomial.get(),
            shell_pair_scale.get()};
  }

  Gfn2ForceDeviceActivity activity(const HostCase& host) const {
    return {requested.get(), statuses.get(), host.basis.batch_size, kPlanToken};
  }

  Gfn2H0ForceDeviceInput input(const HostCase& host) const {
    return {positions.get(),
            host.basis.total_atoms * 3,
            coordination.get(),
            host.basis.total_atoms,
            overlap.get(),
            host.integrals.total_matrix_elements,
            density.get(),
            host.integrals.total_matrix_elements,
            weighted_density.get(),
            host.integrals.total_matrix_elements,
            kPlanToken};
  }

  Gfn2H0ForceDeviceOutput output(const HostCase& host) {
    return {overlap_adjoint.get(),
            host.integrals.total_matrix_elements,
            coordination_adjoint.get(),
            host.basis.total_atoms,
            gradients.get(),
            host.basis.total_atoms * 3,
            kPlanToken};
  }

  Gfn2H0ForceDeviceWorkspace workspace(const HostCase& host) {
    return {overlap_scratch.get(),
            host.integrals.total_matrix_elements,
            coordination_scratch.get(),
            host.basis.total_atoms,
            gradient_scratch.get(),
            host.basis.total_atoms * 3,
            sequence_active.get(),
            1,
            kPlanToken};
  }

  cudaError_t seed(const HostCase& host, cudaStream_t stream) {
    cudaError_t status =
        overlap_adjoint.copy_from(host.overlap_seed.data(), host.overlap_seed.size(), stream);
    if (status == cudaSuccess) {
      status = coordination_adjoint.copy_from(host.coordination_seed.data(),
                                              host.coordination_seed.size(), stream);
    }
    return status == cudaSuccess
               ? gradients.copy_from(host.gradient_seed.data(), host.gradient_seed.size(), stream)
               : status;
  }
};

cudaError_t launch(DeviceFixture& device, const HostCase& host, cudaStream_t stream,
                   std::int64_t linear_tiles_per_system = 1) {
  auto batch = device.batch(host);
  batch.linear_tiles_per_system = linear_tiles_per_system;
  cudaError_t status = xtbloom::detail::cuda::reset_gfn2_h0_force_device_errors_cuda(
      batch.batch_size, device.system_errors.get(), device.device_error.get(), stream);
  if (status != cudaSuccess) {
    return status;
  }
  return xtbloom::detail::cuda::add_gfn2_h0_pulay_gradient_cuda(
      batch, device.plan(host), device.activity(host), device.input(host), device.output(host),
      device.workspace(host), device.system_errors.get(), device.device_error.get(), stream);
}

int download(DeviceFixture& device, const HostCase& host, Expected& actual,
             std::vector<std::uint32_t>& system_errors, std::uint32_t& device_error,
             cudaStream_t stream) {
  actual.overlap_adjoint.resize(host.overlap_seed.size());
  actual.coordination_adjoint.resize(host.coordination_seed.size());
  actual.gradients.resize(host.gradient_seed.size());
  system_errors.resize(static_cast<std::size_t>(host.basis.batch_size));
  CUDA_CHECK(device.overlap_adjoint.copy_to(actual.overlap_adjoint.data(),
                                            actual.overlap_adjoint.size(), stream));
  CUDA_CHECK(device.coordination_adjoint.copy_to(actual.coordination_adjoint.data(),
                                                 actual.coordination_adjoint.size(), stream));
  CUDA_CHECK(device.gradients.copy_to(actual.gradients.data(), actual.gradients.size(), stream));
  CUDA_CHECK(device.system_errors.copy_to(system_errors.data(), system_errors.size(), stream));
  CUDA_CHECK(device.device_error.copy_to(&device_error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return 0;
}

int compare(const Expected& actual, const Expected& expected) {
  CHECK(actual.overlap_adjoint.size() == expected.overlap_adjoint.size());
  CHECK(actual.coordination_adjoint.size() == expected.coordination_adjoint.size());
  CHECK(actual.gradients.size() == expected.gradients.size());
  for (std::size_t index = 0; index < actual.overlap_adjoint.size(); ++index) {
    CHECK(near(actual.overlap_adjoint[index], expected.overlap_adjoint[index]));
  }
  for (std::size_t index = 0; index < actual.coordination_adjoint.size(); ++index) {
    CHECK(near(actual.coordination_adjoint[index], expected.coordination_adjoint[index]));
  }
  for (std::size_t index = 0; index < actual.gradients.size(); ++index) {
    CHECK(near(actual.gradients[index], expected.gradients[index], 8.0e-12, 5.0e-12));
  }
  return 0;
}

int test_cpu_and_finite_difference_parity() {
  HostCase host;
  std::string error;
  CHECK(make_case(host, error));
  Expected expected;
  CHECK(cpu_expected(host, expected, error));

  /* The CUDA output is a gradient; a public force is its exact negation. */
  const std::size_t force_probe = 2u;
  const double expected_force = -expected.gradients[force_probe];
  CHECK(near(expected_force, -expected.gradients[force_probe], 0.0, 0.0));

  constexpr double matrix_step = 1.0e-6;
  std::vector<double> right_overlap = host.overlap;
  std::vector<double> left_overlap = host.overlap;
  right_overlap[1] += matrix_step;
  left_overlap[1] -= matrix_step;
  const double matrix_numerical =
      (stationary_core_energy(host, host.positions, host.coordination, right_overlap, error) -
       stationary_core_energy(host, host.positions, host.coordination, left_overlap, error)) /
      (2.0 * matrix_step);
  CHECK(near(matrix_numerical, expected.overlap_adjoint[1] - host.overlap_seed[1], 3.0e-9, 3.0e-9));

  constexpr double cn_step = 1.0e-6;
  std::vector<double> right_cn = host.coordination;
  std::vector<double> left_cn = host.coordination;
  right_cn[0] += cn_step;
  left_cn[0] -= cn_step;
  const double cn_numerical =
      (stationary_core_energy(host, host.positions, right_cn, host.overlap, error) -
       stationary_core_energy(host, host.positions, left_cn, host.overlap, error)) /
      (2.0 * cn_step);
  CHECK(near(cn_numerical, expected.coordination_adjoint[0] - host.coordination_seed[0], 3.0e-9,
             3.0e-9));

  constexpr double coordinate_step = 2.0e-6;
  std::vector<double> right_positions = host.positions;
  std::vector<double> left_positions = host.positions;
  right_positions[2] += coordinate_step;
  left_positions[2] -= coordinate_step;
  const double coordinate_numerical =
      (stationary_core_energy(host, right_positions, host.coordination, host.overlap, error) -
       stationary_core_energy(host, left_positions, host.coordination, host.overlap, error)) /
      (2.0 * coordinate_step);
  CHECK(near(coordinate_numerical, expected.gradients[2] - host.gradient_seed[2], 2.0e-8, 2.0e-8));

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  const std::vector<std::uint8_t> requested(3u, 1u);
  const std::vector<xtbloom_status_t> statuses(3u, XTBLOOM_STATUS_SUCCESS);
  CUDA_CHECK(device.requested.copy_from(requested.data(), requested.size(), stream));
  CUDA_CHECK(device.statuses.copy_from(statuses.data(), statuses.size(), stream));
  CUDA_CHECK(device.seed(host, stream));
  CUDA_CHECK(launch(device, host, stream));
  Expected actual;
  std::vector<std::uint32_t> system_errors;
  std::uint32_t device_error = 0u;
  CHECK(download(device, host, actual, system_errors, device_error, stream) == 0);
  CHECK(std::all_of(system_errors.begin(), system_errors.end(),
                    [](std::uint32_t value) { return value == 0u; }));
  CHECK(device_error == 0u);
  CHECK(compare(actual, expected) == 0);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

void restore_skipped_slices(const HostCase& host, const std::vector<std::uint8_t>& requested,
                            const std::vector<xtbloom_status_t>& statuses, Expected& expected) {
  for (std::int64_t system = 0; system < host.basis.batch_size; ++system) {
    const std::size_t index = static_cast<std::size_t>(system);
    if (requested[index] == 1u && statuses[index] == XTBLOOM_STATUS_SUCCESS) {
      continue;
    }
    const std::int64_t matrix_begin = host.integrals.matrix_offsets[index];
    const std::int64_t matrix_end = host.integrals.matrix_offsets[index + 1u];
    std::copy(host.overlap_seed.begin() + matrix_begin, host.overlap_seed.begin() + matrix_end,
              expected.overlap_adjoint.begin() + matrix_begin);
    const std::int64_t atom_begin = host.basis.atom_offsets[index];
    const std::int64_t atom_end = host.basis.atom_offsets[index + 1u];
    std::copy(host.coordination_seed.begin() + atom_begin,
              host.coordination_seed.begin() + atom_end,
              expected.coordination_adjoint.begin() + atom_begin);
    std::copy(host.gradient_seed.begin() + 3 * atom_begin,
              host.gradient_seed.begin() + 3 * atom_end,
              expected.gradients.begin() + 3 * atom_begin);
  }
}

int test_activity_status_gate_and_transactionality() {
  HostCase host;
  std::string error;
  CHECK(make_case(host, error));
  Expected expected;
  CHECK(cpu_expected(host, expected, error));
  const std::vector<std::uint8_t> requested{1u, 0u, 1u};
  const std::vector<xtbloom_status_t> statuses{XTBLOOM_STATUS_SUCCESS, XTBLOOM_STATUS_SUCCESS,
                                               XTBLOOM_STATUS_SCC_NOT_CONVERGED};
  restore_skipped_slices(host, requested, statuses, expected);

  std::vector<double> poisoned_density = host.density;
  for (std::size_t system : {1u, 2u}) {
    const std::int64_t begin = host.integrals.matrix_offsets[system];
    const std::int64_t end = host.integrals.matrix_offsets[system + 1u];
    std::fill(poisoned_density.begin() + begin, poisoned_density.begin() + end,
              std::numeric_limits<double>::quiet_NaN());
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CUDA_CHECK(device.density.copy_from(poisoned_density.data(), poisoned_density.size(), stream));
  CUDA_CHECK(device.requested.copy_from(requested.data(), requested.size(), stream));
  CUDA_CHECK(device.statuses.copy_from(statuses.data(), statuses.size(), stream));
  CUDA_CHECK(device.seed(host, stream));
  CUDA_CHECK(launch(device, host, stream));
  Expected actual;
  std::vector<std::uint32_t> system_errors;
  std::uint32_t device_error = 0u;
  CHECK(download(device, host, actual, system_errors, device_error, stream) == 0);
  CHECK(std::all_of(system_errors.begin(), system_errors.end(),
                    [](std::uint32_t value) { return value == 0u; }));
  CHECK(device_error == 0u);
  CHECK(compare(actual, expected) == 0);

  /* Invalid activity is a peer failure and retains every seeded output byte. */
  const std::vector<std::uint8_t> invalid_requested{2u, 0u, 0u};
  CUDA_CHECK(
      device.requested.copy_from(invalid_requested.data(), invalid_requested.size(), stream));
  CUDA_CHECK(device.seed(host, stream));
  CUDA_CHECK(launch(device, host, stream));
  CHECK(download(device, host, actual, system_errors, device_error, stream) == 0);
  CHECK(system_errors[0] == static_cast<std::uint32_t>(Gfn2H0ForceDeviceError::kInvalidActiveMask));
  CHECK(system_errors[1] == 0u && system_errors[2] == 0u);
  CHECK(device_error == static_cast<std::uint32_t>(Gfn2H0ForceDeviceError::kInvalidActiveMask));
  Expected seeds{host.overlap_seed, host.coordination_seed, host.gradient_seed};
  CHECK(compare(actual, seeds) == 0);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_cuda_graph_capture() {
  HostCase host;
  std::string error;
  CHECK(make_case(host, error));
  Expected expected;
  CHECK(cpu_expected(host, expected, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  const std::vector<std::uint8_t> requested(3u, 1u);
  const std::vector<xtbloom_status_t> statuses(3u, XTBLOOM_STATUS_SUCCESS);
  CUDA_CHECK(device.requested.copy_from(requested.data(), requested.size(), stream));
  CUDA_CHECK(device.statuses.copy_from(statuses.data(), statuses.size(), stream));
  CUDA_CHECK(device.seed(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(launch(device, host, stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  Expected actual;
  std::vector<std::uint32_t> system_errors;
  std::uint32_t device_error = 0u;
  CHECK(download(device, host, actual, system_errors, device_error, stream) == 0);
  CHECK(device_error == 0u);
  CHECK(compare(actual, expected) == 0);
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_large_singleton_tiling_graph_and_late_failure() {
  HostCase host;
  std::string error;
  CHECK(make_case(host, error, 129u));
  CHECK(host.basis.total_atoms == 129);
  std::fill(host.density.begin(), host.density.end(), 0.0);
  const std::array<std::int32_t, 1> spin_channels{1};
  std::int64_t tiles = 0;
  CHECK(select_gfn2_density_contraction_tiles(
      host.basis.batch_orbital_offsets.data(), host.basis.batch_orbital_offsets.size(),
      spin_channels.data(), spin_channels.size(), host.basis.batch_size, tiles));
  CHECK(tiles > 1 && tiles <= kGfn2IntegralLinearBlockBudget);
  CHECK(host.integrals.total_matrix_elements > (tiles - 1) * 128);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  const std::vector<std::uint8_t> requested(1u, 1u);
  const std::vector<xtbloom_status_t> statuses(1u, XTBLOOM_STATUS_SUCCESS);
  CUDA_CHECK(device.requested.copy_from(requested.data(), requested.size(), stream));
  CUDA_CHECK(device.statuses.copy_from(statuses.data(), statuses.size(), stream));

  CUDA_CHECK(device.seed(host, stream));
  CUDA_CHECK(launch(device, host, stream, 1));
  Expected single;
  std::vector<std::uint32_t> system_errors;
  std::uint32_t device_error = 0u;
  CHECK(download(device, host, single, system_errors, device_error, stream) == 0);
  CHECK(device_error == 0u && system_errors[0] == 0u);

  CUDA_CHECK(device.seed(host, stream));
  CUDA_CHECK(launch(device, host, stream, tiles));
  Expected tiled;
  CHECK(download(device, host, tiled, system_errors, device_error, stream) == 0);
  CHECK(device_error == 0u && system_errors[0] == 0u);
  CHECK(tiled.overlap_adjoint == single.overlap_adjoint);
  CHECK(tiled.coordination_adjoint == single.coordination_adjoint);
  CHECK(tiled.gradients == single.gradients);

  CUDA_CHECK(device.seed(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  Gfn2IntegralDeviceBatch tiled_batch = device.batch(host);
  tiled_batch.linear_tiles_per_system = tiles;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_h0_force_device_errors_cuda(
      tiled_batch.batch_size, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_h0_pulay_gradient_cuda(
      tiled_batch, device.plan(host), device.activity(host), device.input(host),
      device.output(host), device.workspace(host), device.system_errors.get(),
      device.device_error.get(), stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  std::size_t tiled_nodes = 0u;
  CHECK(count_kernel_grid(graph, dim3(1u, static_cast<unsigned int>(tiles), 1u), tiled_nodes) == 0);
  /* Seed/preflight and final publication share the topology-fixed grid. */
  CHECK(tiled_nodes == 2u);
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CHECK(download(device, host, tiled, system_errors, device_error, stream) == 0);
  CHECK(tiled.overlap_adjoint == single.overlap_adjoint);
  CHECK(tiled.coordination_adjoint == single.coordination_adjoint);
  CHECK(tiled.gradients == single.gradients);
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));

  Gfn2IntegralDeviceBatch invalid_batch = device.batch(host);
  invalid_batch.linear_tiles_per_system = 0;
  CHECK(xtbloom::detail::cuda::add_gfn2_h0_pulay_gradient_cuda(
            invalid_batch, device.plan(host), device.activity(host), device.input(host),
            device.output(host), device.workspace(host), device.system_errors.get(),
            device.device_error.get(), stream) == cudaErrorInvalidValue);
  invalid_batch.linear_tiles_per_system = kGfn2IntegralLinearBlockBudget + 1;
  CHECK(xtbloom::detail::cuda::add_gfn2_h0_pulay_gradient_cuda(
            invalid_batch, device.plan(host), device.activity(host), device.input(host),
            device.output(host), device.workspace(host), device.system_errors.get(),
            device.device_error.get(), stream) == cudaErrorInvalidValue);

  CUDA_CHECK(device.seed(host, stream));
  const std::int64_t late_element = (tiles - 1) * 128;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(cudaMemcpyAsync(device.weighted_density.get() + late_element, &nan, sizeof(nan),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(launch(device, host, stream, tiles));
  CHECK(download(device, host, tiled, system_errors, device_error, stream) == 0);
  CHECK(system_errors[0] == static_cast<std::uint32_t>(Gfn2H0ForceDeviceError::kNonfiniteInput));
  CHECK(device_error == static_cast<std::uint32_t>(Gfn2H0ForceDeviceError::kNonfiniteInput));
  CHECK(tiled.overlap_adjoint == host.overlap_seed);
  CHECK(tiled.coordination_adjoint == host.coordination_seed);
  CHECK(tiled.gradients == host.gradient_seed);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_cpu_and_finite_difference_parity(); status != 0) {
    return status;
  }
  if (const int status = test_activity_status_gate_and_transactionality(); status != 0) {
    return status;
  }
  if (const int status = test_cuda_graph_capture(); status != 0) {
    return status;
  }
  return test_large_singleton_tiling_graph_and_late_failure();
}
