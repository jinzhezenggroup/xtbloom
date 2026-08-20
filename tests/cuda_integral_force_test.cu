#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_density.cuh"
#include "backends/cuda/gfn2_integral_force.cuh"
#include "model/gfn2/basis.hpp"
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
using xtbloom::detail::cuda::Gfn2IntegralDeviceBatch;
using xtbloom::detail::cuda::Gfn2IntegralDeviceError;
using xtbloom::detail::cuda::Gfn2IntegralForceDeviceInput;
using xtbloom::detail::cuda::Gfn2IntegralForceDeviceOutput;
using xtbloom::detail::cuda::Gfn2IntegralForceDeviceWorkspace;
using xtbloom::detail::cuda::kGfn2IntegralLinearBlockBudget;
using xtbloom::detail::cuda::select_gfn2_density_contraction_tiles;
using xtbloom::detail::gfn2::BasisPlan;
using xtbloom::detail::gfn2::IntegralPlan;

constexpr std::uint64_t kPlanToken = 0x2b41602af7c913d5ULL;

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

  cudaError_t copy_from(const T* source, std::size_t count, cudaStream_t stream) {
    if (source == nullptr || count > count_) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(data_, source, count * sizeof(T), cudaMemcpyHostToDevice, stream);
  }

  cudaError_t copy_to(T* destination, std::size_t count, cudaStream_t stream) const {
    if (destination == nullptr || count > count_) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
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

bool near(double actual, double expected, double absolute = 2.0e-10, double relative = 2.0e-10) {
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
  std::vector<std::int64_t> shell_pair_offsets;
  std::int64_t maximum_system_shells = 0;
  std::vector<double> positions;
  std::vector<double> overlap_adjoint;
  std::vector<double> dipole_adjoint;
  std::vector<double> quadrupole_adjoint;
  std::vector<double> seed;
  std::vector<double> reference;
};

void append_system(std::size_t system, std::vector<std::int32_t>& atomic_numbers,
                   std::vector<double>& positions) {
  const double origin = 4.5 * static_cast<double>(system);
  auto append_atom = [&](std::int32_t number, double x, double y, double z) {
    atomic_numbers.push_back(number);
    positions.push_back(origin + x);
    positions.push_back(y);
    positions.push_back(z);
  };
  if (system == 0u) {
    /* Silicon exercises every supported s/p/d shell derivative path. */
    append_atom(14, 0.1, -0.2, 0.3);
    append_atom(14, 1.4, -0.8, 2.1);
  } else if (system == 1u) {
    append_atom(1, 0.0, 0.0, 0.0);
    append_atom(1, 0.8, 0.2, -0.4);
  } else if (system == 2u) {
    append_atom(8, -0.2, 0.3, 0.1);
    append_atom(1, 1.1, -0.5, 0.7);
  } else if (system % 11u == 0u) {
    /* Empty members ensure all descriptors are genuinely ragged. */
  } else if (system % 5u == 0u) {
    append_atom(6, -0.1, 0.2, -0.3);
    append_atom(1, 0.9, -0.4, 0.5);
  } else {
    append_atom(system % 2u == 0u ? 8 : 1, 0.01 * static_cast<double>(system % 7u),
                -0.02 * static_cast<double>(system % 5u), 0.03 * static_cast<double>(system % 3u));
  }
}

bool update_reference(HostCase& data, std::string& error) {
  data.reference = data.seed;
  std::vector<double> workspace((data.integrals.workspace_size_bytes + sizeof(double) - 1u) /
                                sizeof(double));
  return xtbloom::detail::gfn2::add_overlap_gradient_cpu(
             data.basis, data.integrals, data.positions.data(), data.overlap_adjoint.data(),
             data.reference.data(), workspace.data(), workspace.size() * sizeof(double),
             error) == XTBLOOM_STATUS_SUCCESS &&
         xtbloom::detail::gfn2::add_multipole_gradient_cpu(
             data.basis, data.integrals, data.positions.data(), data.dipole_adjoint.data(),
             data.quadrupole_adjoint.data(), data.reference.data(), workspace.data(),
             workspace.size() * sizeof(double), error) == XTBLOOM_STATUS_SUCCESS;
}

bool make_case(std::size_t batch_size, HostCase& data, std::string& error,
               std::size_t large_system_atoms = 0u) {
  std::vector<std::int64_t> atom_offsets(batch_size + 1u, 0);
  std::vector<std::int32_t> atomic_numbers;
  for (std::size_t system = 0; system < batch_size; ++system) {
    atom_offsets[system] = static_cast<std::int64_t>(atomic_numbers.size());
    if (large_system_atoms != 0u) {
      for (std::size_t atom = 0; atom < large_system_atoms; ++atom) {
        atomic_numbers.push_back(1);
        /* Keep large ragged peers spatially separated while ensuring every
         * system has enough matrix work to exercise more than one tile. */
        data.positions.push_back(200.0 * static_cast<double>(system) +
                                 2.25 * static_cast<double>(atom));
        data.positions.push_back(0.17 * static_cast<double>(atom % 3u));
        data.positions.push_back(-0.11 * static_cast<double>(atom % 5u));
      }
    } else {
      append_system(system, atomic_numbers, data.positions);
    }
  }
  atom_offsets[batch_size] = static_cast<std::int64_t>(atomic_numbers.size());
  if (xtbloom::detail::gfn2::make_basis_plan(static_cast<std::int64_t>(batch_size),
                                             static_cast<std::int64_t>(atomic_numbers.size()),
                                             atom_offsets.data(), atomic_numbers.data(), data.basis,
                                             error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_integral_plan(data.basis, data.integrals, error) !=
          XTBLOOM_STATUS_SUCCESS) {
    return false;
  }

  data.shell_pair_offsets.resize(batch_size + 1u, 0);
  for (std::size_t system = 0; system < batch_size; ++system) {
    const std::int64_t shells =
        data.basis.batch_shell_offsets[system + 1u] - data.basis.batch_shell_offsets[system];
    data.maximum_system_shells = std::max(data.maximum_system_shells, shells);
    data.shell_pair_offsets[system + 1u] = data.shell_pair_offsets[system] + shells * shells;
  }

  const std::size_t matrices = static_cast<std::size_t>(data.integrals.total_matrix_elements);
  const std::size_t coordinates = static_cast<std::size_t>(data.basis.total_atoms) * 3u;
  data.overlap_adjoint.resize(matrices);
  data.dipole_adjoint.resize(3u * matrices);
  data.quadrupole_adjoint.resize(6u * matrices);
  data.seed.resize(coordinates);
  for (std::size_t element = 0; element < matrices; ++element) {
    data.overlap_adjoint[element] = 0.37 * std::sin(0.17 * static_cast<double>(element + 1u)) +
                                    0.11 * std::cos(0.31 * static_cast<double>(element + 3u));
  }
  for (std::size_t element = 0; element < data.dipole_adjoint.size(); ++element) {
    data.dipole_adjoint[element] = 0.23 * std::sin(0.13 * static_cast<double>(element + 2u)) -
                                   0.07 * std::cos(0.29 * static_cast<double>(element + 5u));
  }
  for (std::size_t element = 0; element < data.quadrupole_adjoint.size(); ++element) {
    data.quadrupole_adjoint[element] = 0.19 * std::cos(0.09 * static_cast<double>(element + 4u)) +
                                       0.05 * std::sin(0.27 * static_cast<double>(element + 7u));
  }
  for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate) {
    data.seed[coordinate] = 0.04 * std::sin(0.21 * static_cast<double>(coordinate + 1u));
  }
  return update_reference(data, error);
}

struct DeviceFixture {
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> batch_shell_offsets;
  DeviceBuffer<std::int64_t> batch_orbital_offsets;
  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<std::int64_t> shell_pair_offsets;
  DeviceBuffer<std::int64_t> atom_shell_offsets;
  DeviceBuffer<std::int64_t> shell_orbital_offsets;
  DeviceBuffer<std::int64_t> shell_primitive_offsets;
  DeviceBuffer<std::int64_t> shell_to_atom;
  DeviceBuffer<std::uint8_t> angular_momenta;
  DeviceBuffer<double> primitive_exponents;
  DeviceBuffer<double> primitive_coefficients;
  DeviceBuffer<double> positions;
  DeviceBuffer<double> overlap_adjoint;
  DeviceBuffer<double> dipole_adjoint;
  DeviceBuffer<double> quadrupole_adjoint;
  DeviceBuffer<double> gradients;
  DeviceBuffer<double> gradient_scratch;
  DeviceBuffer<std::uint8_t> requested;
  DeviceBuffer<xtbloom_status_t> statuses;
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
    UPLOAD(shell_pair_offsets, host.shell_pair_offsets)
    UPLOAD(atom_shell_offsets, host.basis.atom_shell_offsets)
    UPLOAD(shell_orbital_offsets, host.basis.shell_orbital_offsets)
    UPLOAD(shell_primitive_offsets, host.basis.shell_primitive_offsets)
    UPLOAD(shell_to_atom, host.basis.shell_to_atom)
    UPLOAD(angular_momenta, host.basis.angular_momenta)
    UPLOAD(primitive_exponents, host.basis.primitive_exponents)
    UPLOAD(primitive_coefficients, host.basis.primitive_coefficients)
#undef UPLOAD
    const std::size_t coordinates = host.positions.size();
    const std::size_t systems = static_cast<std::size_t>(host.basis.batch_size);
    if (status == cudaSuccess) {
      status = positions.allocate(coordinates);
    }
    if (status == cudaSuccess) {
      status = overlap_adjoint.allocate(host.overlap_adjoint.size());
    }
    if (status == cudaSuccess) {
      status = dipole_adjoint.allocate(host.dipole_adjoint.size());
    }
    if (status == cudaSuccess) {
      status = quadrupole_adjoint.allocate(host.quadrupole_adjoint.size());
    }
    if (status == cudaSuccess) {
      status = gradients.allocate(coordinates);
    }
    if (status == cudaSuccess) {
      status = gradient_scratch.allocate(coordinates);
    }
    if (status == cudaSuccess) {
      status = requested.allocate(systems);
    }
    if (status == cudaSuccess) {
      status = statuses.allocate(systems);
    }
    if (status == cudaSuccess) {
      status = sequence_active.allocate(1u);
    }
    if (status == cudaSuccess) {
      status = system_errors.allocate(systems);
    }
    if (status == cudaSuccess) {
      status = device_error.allocate(1u);
    }
    if (status != cudaSuccess) {
      return status;
    }
    std::vector<std::uint8_t> all_requested(systems, 1u);
    std::vector<xtbloom_status_t> all_success(systems, XTBLOOM_STATUS_SUCCESS);
    status = requested.copy_from(all_requested.data(), systems, stream);
    if (status == cudaSuccess) {
      status = statuses.copy_from(all_success.data(), systems, stream);
    }
    return status == cudaSuccess ? upload_dynamic(host, stream) : status;
  }

  cudaError_t upload_dynamic(const HostCase& host, cudaStream_t stream) {
    cudaError_t status = positions.copy_from(host.positions.data(), host.positions.size(), stream);
    if (status == cudaSuccess) {
      status = overlap_adjoint.copy_from(host.overlap_adjoint.data(), host.overlap_adjoint.size(),
                                         stream);
    }
    if (status == cudaSuccess) {
      status =
          dipole_adjoint.copy_from(host.dipole_adjoint.data(), host.dipole_adjoint.size(), stream);
    }
    if (status == cudaSuccess) {
      status = quadrupole_adjoint.copy_from(host.quadrupole_adjoint.data(),
                                            host.quadrupole_adjoint.size(), stream);
    }
    return status == cudaSuccess ? gradients.copy_from(host.seed.data(), host.seed.size(), stream)
                                 : status;
  }

  Gfn2IntegralDeviceBatch batch(const HostCase& host) const {
    return Gfn2IntegralDeviceBatch{
        host.basis.batch_size,
        host.basis.total_atoms,
        host.basis.total_shells,
        host.basis.total_orbitals,
        host.basis.total_primitives,
        host.integrals.total_matrix_elements,
        host.shell_pair_offsets.back(),
        host.maximum_system_shells,
        1,
        host.integrals.integral_cutoff,
        kPlanToken,
        static_cast<std::int64_t>(host.basis.atom_offsets.size()),
        static_cast<std::int64_t>(host.basis.batch_shell_offsets.size()),
        static_cast<std::int64_t>(host.basis.batch_orbital_offsets.size()),
        static_cast<std::int64_t>(host.integrals.matrix_offsets.size()),
        static_cast<std::int64_t>(host.shell_pair_offsets.size()),
        static_cast<std::int64_t>(host.basis.atom_shell_offsets.size()),
        static_cast<std::int64_t>(host.basis.shell_orbital_offsets.size()),
        static_cast<std::int64_t>(host.basis.shell_primitive_offsets.size()),
        static_cast<std::int64_t>(host.basis.shell_to_atom.size()),
        static_cast<std::int64_t>(host.basis.angular_momenta.size()),
        static_cast<std::int64_t>(host.basis.primitive_exponents.size()),
        static_cast<std::int64_t>(host.basis.primitive_coefficients.size()),
        atom_offsets.get(),
        batch_shell_offsets.get(),
        batch_orbital_offsets.get(),
        matrix_offsets.get(),
        shell_pair_offsets.get(),
        atom_shell_offsets.get(),
        shell_orbital_offsets.get(),
        shell_primitive_offsets.get(),
        shell_to_atom.get(),
        angular_momenta.get(),
        primitive_exponents.get(),
        primitive_coefficients.get()};
  }

  Gfn2ForceDeviceActivity activity(const HostCase& host) const {
    return Gfn2ForceDeviceActivity{requested.get(), statuses.get(), host.basis.batch_size,
                                   kPlanToken};
  }

  Gfn2IntegralForceDeviceInput input(const HostCase& host) const {
    return Gfn2IntegralForceDeviceInput{positions.get(),
                                        static_cast<std::int64_t>(host.positions.size()),
                                        overlap_adjoint.get(),
                                        static_cast<std::int64_t>(host.overlap_adjoint.size()),
                                        dipole_adjoint.get(),
                                        static_cast<std::int64_t>(host.dipole_adjoint.size()),
                                        quadrupole_adjoint.get(),
                                        static_cast<std::int64_t>(host.quadrupole_adjoint.size()),
                                        kPlanToken};
  }

  Gfn2IntegralForceDeviceOutput output(const HostCase& host) {
    return Gfn2IntegralForceDeviceOutput{
        gradients.get(), static_cast<std::int64_t>(host.positions.size()), kPlanToken};
  }

  Gfn2IntegralForceDeviceWorkspace workspace(const HostCase& host) {
    return Gfn2IntegralForceDeviceWorkspace{gradient_scratch.get(),
                                            static_cast<std::int64_t>(host.positions.size()),
                                            sequence_active.get(), 1, kPlanToken};
  }
};

int run_force(DeviceFixture& device, const HostCase& host, cudaStream_t stream,
              std::int64_t linear_tiles_per_system = 1) {
  Gfn2IntegralDeviceBatch batch = device.batch(host);
  batch.linear_tiles_per_system = linear_tiles_per_system;
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_integral_force_device_errors_cuda(
      batch.batch_size, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_integral_gradient_cuda(
      batch, device.activity(host), device.input(host), device.output(host), device.workspace(host),
      device.system_errors.get(), device.device_error.get(), stream));
  return 0;
}

int check_result(DeviceFixture& device, const HostCase& host, const std::vector<double>& expected,
                 cudaStream_t stream) {
  std::vector<double> actual(expected.size());
  std::vector<std::uint32_t> errors(static_cast<std::size_t>(host.basis.batch_size));
  std::uint32_t diagnostic = 0u;
  CUDA_CHECK(device.gradients.copy_to(actual.data(), actual.size(), stream));
  CUDA_CHECK(device.system_errors.copy_to(errors.data(), errors.size(), stream));
  CUDA_CHECK(device.device_error.copy_to(&diagnostic, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(diagnostic == static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kSuccess));
  CHECK(std::all_of(errors.begin(), errors.end(), [](std::uint32_t value) {
    return value == static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kSuccess);
  }));
  for (std::size_t coordinate = 0; coordinate < actual.size(); ++coordinate) {
    CHECK(near(actual[coordinate], expected[coordinate]));
  }
  return 0;
}

int test_cpu_parity_seed_and_ragged_batches() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (std::size_t batch_size : {1u, 8u, 32u, 128u}) {
    HostCase host;
    std::string error;
    CHECK(make_case(batch_size, host, error));
    DeviceFixture device;
    CUDA_CHECK(device.initialize(host, stream));
    CHECK(run_force(device, host, stream) == 0);
    CHECK(check_result(device, host, host.reference, stream) == 0);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

enum class AdjointKind {
  kOverlap,
  kDipole,
  kQuadrupole,
};

double weighted_integral_value(const HostCase& host, const std::vector<double>& positions,
                               std::string& error) {
  const std::size_t matrices = static_cast<std::size_t>(host.integrals.total_matrix_elements);
  std::vector<double> overlap(matrices);
  std::vector<double> dipole(3u * matrices);
  std::vector<double> quadrupole(6u * matrices);
  std::vector<double> workspace((host.integrals.workspace_size_bytes + sizeof(double) - 1u) /
                                sizeof(double));
  if (xtbloom::detail::gfn2::evaluate_overlap_cpu(
          host.basis, host.integrals, positions.data(), overlap.data(), workspace.data(),
          workspace.size() * sizeof(double), error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::evaluate_multipole_cpu(
          host.basis, host.integrals, positions.data(), dipole.data(), quadrupole.data(),
          workspace.data(), workspace.size() * sizeof(double), error) != XTBLOOM_STATUS_SUCCESS) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double value = 0.0;
  for (std::size_t element = 0; element < matrices; ++element) {
    value += host.overlap_adjoint[element] * overlap[element];
  }
  for (std::size_t element = 0; element < dipole.size(); ++element) {
    value += host.dipole_adjoint[element] * dipole[element];
  }
  for (std::size_t element = 0; element < quadrupole.size(); ++element) {
    value += host.quadrupole_adjoint[element] * quadrupole[element];
  }
  return value;
}

int test_component_finite_differences() {
  for (AdjointKind kind : {AdjointKind::kOverlap, AdjointKind::kDipole, AdjointKind::kQuadrupole}) {
    HostCase host;
    std::string error;
    CHECK(make_case(1u, host, error));
    std::fill(host.seed.begin(), host.seed.end(), 0.0);
    if (kind != AdjointKind::kOverlap) {
      std::fill(host.overlap_adjoint.begin(), host.overlap_adjoint.end(), 0.0);
    }
    if (kind != AdjointKind::kDipole) {
      std::fill(host.dipole_adjoint.begin(), host.dipole_adjoint.end(), 0.0);
    }
    if (kind != AdjointKind::kQuadrupole) {
      std::fill(host.quadrupole_adjoint.begin(), host.quadrupole_adjoint.end(), 0.0);
    }
    CHECK(update_reference(host, error));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    DeviceFixture device;
    CUDA_CHECK(device.initialize(host, stream));
    CHECK(run_force(device, host, stream) == 0);
    CHECK(check_result(device, host, host.reference, stream) == 0);

    constexpr double step = 2.0e-5;
    for (std::size_t coordinate = 0; coordinate < host.positions.size(); ++coordinate) {
      std::vector<double> displaced = host.positions;
      displaced[coordinate] += step;
      const double right = weighted_integral_value(host, displaced, error);
      displaced[coordinate] -= 2.0 * step;
      const double left = weighted_integral_value(host, displaced, error);
      CHECK(std::isfinite(right));
      CHECK(std::isfinite(left));
      const double numerical = (right - left) / (2.0 * step);
      CHECK(near(host.reference[coordinate], numerical, 2.0e-8, 2.0e-8));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
  return 0;
}

int test_requested_and_failed_poison_peers() {
  HostCase host;
  std::string error;
  CHECK(make_case(8u, host, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));

  constexpr std::size_t unrequested = 1u;
  constexpr std::size_t failed = 2u;
  std::vector<std::uint8_t> requested(8u, 1u);
  std::vector<xtbloom_status_t> statuses(8u, XTBLOOM_STATUS_SUCCESS);
  requested[unrequested] = 0u;
  statuses[failed] = XTBLOOM_STATUS_INTERNAL_ERROR;
  CUDA_CHECK(device.requested.copy_from(requested.data(), requested.size(), stream));
  CUDA_CHECK(device.statuses.copy_from(statuses.data(), statuses.size(), stream));

  HostCase poisoned = host;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t system : {unrequested, failed}) {
    const std::size_t atom_begin = static_cast<std::size_t>(host.basis.atom_offsets[system]);
    const std::size_t atom_end = static_cast<std::size_t>(host.basis.atom_offsets[system + 1u]);
    std::fill(poisoned.positions.begin() + static_cast<std::ptrdiff_t>(atom_begin * 3u),
              poisoned.positions.begin() + static_cast<std::ptrdiff_t>(atom_end * 3u), nan);
    const std::size_t matrix_begin =
        static_cast<std::size_t>(host.integrals.matrix_offsets[system]);
    const std::size_t matrix_end =
        static_cast<std::size_t>(host.integrals.matrix_offsets[system + 1u]);
    std::fill(poisoned.overlap_adjoint.begin() + static_cast<std::ptrdiff_t>(matrix_begin),
              poisoned.overlap_adjoint.begin() + static_cast<std::ptrdiff_t>(matrix_end), nan);
    for (std::size_t component = 0; component < 3u; ++component) {
      std::fill(
          poisoned.dipole_adjoint.begin() +
              static_cast<std::ptrdiff_t>(component * host.overlap_adjoint.size() + matrix_begin),
          poisoned.dipole_adjoint.begin() +
              static_cast<std::ptrdiff_t>(component * host.overlap_adjoint.size() + matrix_end),
          nan);
    }
    for (std::size_t component = 0; component < 6u; ++component) {
      std::fill(
          poisoned.quadrupole_adjoint.begin() +
              static_cast<std::ptrdiff_t>(component * host.overlap_adjoint.size() + matrix_begin),
          poisoned.quadrupole_adjoint.begin() +
              static_cast<std::ptrdiff_t>(component * host.overlap_adjoint.size() + matrix_end),
          nan);
    }
  }
  CUDA_CHECK(device.upload_dynamic(poisoned, stream));
  CHECK(run_force(device, poisoned, stream) == 0);

  std::vector<double> expected = host.reference;
  for (std::size_t system : {unrequested, failed}) {
    const std::size_t begin = static_cast<std::size_t>(host.basis.atom_offsets[system]) * 3u;
    const std::size_t end = static_cast<std::size_t>(host.basis.atom_offsets[system + 1u]) * 3u;
    std::copy(host.seed.begin() + static_cast<std::ptrdiff_t>(begin),
              host.seed.begin() + static_cast<std::ptrdiff_t>(end),
              expected.begin() + static_cast<std::ptrdiff_t>(begin));
  }
  CHECK(check_result(device, poisoned, expected, stream) == 0);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_changed_input_graph_replay() {
  HostCase host;
  std::string error;
  CHECK(make_case(8u, host, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const Gfn2IntegralDeviceBatch batch = device.batch(host);
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_integral_force_device_errors_cuda(
      batch.batch_size, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_integral_gradient_cuda(
      batch, device.activity(host), device.input(host), device.output(host), device.workspace(host),
      device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));

  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CHECK(check_result(device, host, host.reference, stream) == 0);

  host.positions[0] += 0.17;
  host.positions[4] -= 0.09;
  for (std::size_t element = 0; element < host.overlap_adjoint.size(); ++element) {
    host.overlap_adjoint[element] += 0.03 * std::sin(static_cast<double>(element + 1u));
  }
  for (std::size_t element = 0; element < host.dipole_adjoint.size(); ++element) {
    host.dipole_adjoint[element] *= 0.83;
  }
  for (std::size_t coordinate = 0; coordinate < host.seed.size(); ++coordinate) {
    host.seed[coordinate] = -0.02 * std::cos(0.19 * static_cast<double>(coordinate + 2u));
  }
  CHECK(update_reference(host, error));
  CUDA_CHECK(device.upload_dynamic(host, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CHECK(check_result(device, host, host.reference, stream) == 0);

  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_large_singleton_tiling_graph_and_late_failure() {
  HostCase host;
  std::string error;
  CHECK(make_case(1u, host, error, 65u));
  CHECK(host.basis.total_atoms == 65);
  const std::array<std::int32_t, 1> spin_channels{1};
  std::int64_t tiles = 0;
  CHECK(select_gfn2_density_contraction_tiles(
      host.basis.batch_orbital_offsets.data(), host.basis.batch_orbital_offsets.size(),
      spin_channels.data(), spin_channels.size(), host.basis.batch_size, tiles));
  CHECK(tiles > 1 && tiles <= kGfn2IntegralLinearBlockBudget);
  CHECK(host.integrals.total_matrix_elements > (tiles - 1) * 64);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));

  CUDA_CHECK(device.upload_dynamic(host, stream));
  CHECK(run_force(device, host, stream, 1) == 0);
  std::vector<double> single(host.seed.size());
  CUDA_CHECK(device.gradients.copy_to(single.data(), single.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(std::equal(single.begin(), single.end(), host.reference.begin(),
                   [](double actual, double expected) { return near(actual, expected); }));

  CUDA_CHECK(device.upload_dynamic(host, stream));
  CHECK(run_force(device, host, stream, tiles) == 0);
  std::vector<double> tiled(host.seed.size());
  CUDA_CHECK(device.gradients.copy_to(tiled.data(), tiled.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(std::equal(tiled.begin(), tiled.end(), single.begin(),
                   [](double actual, double expected) { return near(actual, expected); }));

  CUDA_CHECK(device.upload_dynamic(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  Gfn2IntegralDeviceBatch tiled_batch = device.batch(host);
  tiled_batch.linear_tiles_per_system = tiles;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_integral_force_device_errors_cuda(
      tiled_batch.batch_size, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_integral_gradient_cuda(
      tiled_batch, device.activity(host), device.input(host), device.output(host),
      device.workspace(host), device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  std::size_t tiled_nodes = 0u;
  CHECK(count_kernel_grid(graph, dim3(1u, static_cast<unsigned int>(tiles), 1u), tiled_nodes) == 0);
  /* Preflight and final gradient publication share the topology-fixed grid. */
  CHECK(tiled_nodes == 2u);
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(device.gradients.copy_to(tiled.data(), tiled.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(std::equal(tiled.begin(), tiled.end(), single.begin(),
                   [](double actual, double expected) { return near(actual, expected); }));
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));

  Gfn2IntegralDeviceBatch invalid_batch = device.batch(host);
  const std::vector<double> invalid_gradients(host.seed.size(), -91.25);
  const std::vector<std::uint32_t> invalid_system_errors(1u, 0xa5a5a5a5u);
  constexpr std::uint32_t kInvalidDeviceErrorSentinel = 0x5a5a5a5au;
  auto check_invalid_tiles_transactional = [&](std::int64_t invalid_tiles) -> int {
    /* Admission failures happen before any launch, so all caller-visible
     * buffers must retain their deliberately non-default sentinels. */
    CUDA_CHECK(
        device.gradients.copy_from(invalid_gradients.data(), invalid_gradients.size(), stream));
    CUDA_CHECK(device.system_errors.copy_from(invalid_system_errors.data(),
                                              invalid_system_errors.size(), stream));
    CUDA_CHECK(device.device_error.copy_from(&kInvalidDeviceErrorSentinel, 1u, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    invalid_batch.linear_tiles_per_system = invalid_tiles;
    CHECK(xtbloom::detail::cuda::add_gfn2_integral_gradient_cuda(
              invalid_batch, device.activity(host), device.input(host), device.output(host),
              device.workspace(host), device.system_errors.get(), device.device_error.get(),
              stream) == cudaErrorInvalidValue);

    std::vector<double> actual_gradients(invalid_gradients.size());
    std::vector<std::uint32_t> actual_system_errors(invalid_system_errors.size());
    std::uint32_t actual_device_error = 0u;
    CUDA_CHECK(device.gradients.copy_to(actual_gradients.data(), actual_gradients.size(), stream));
    CUDA_CHECK(device.system_errors.copy_to(actual_system_errors.data(),
                                            actual_system_errors.size(), stream));
    CUDA_CHECK(device.device_error.copy_to(&actual_device_error, 1u, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(actual_gradients == invalid_gradients);
    CHECK(actual_system_errors == invalid_system_errors);
    CHECK(actual_device_error == kInvalidDeviceErrorSentinel);
    return 0;
  };
  CHECK(check_invalid_tiles_transactional(-1) == 0);
  CHECK(check_invalid_tiles_transactional(0) == 0);
  CHECK(check_invalid_tiles_transactional(kGfn2IntegralLinearBlockBudget + 1) == 0);

  CUDA_CHECK(device.upload_dynamic(host, stream));
  const std::int64_t late_element = (tiles - 1) * 64;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(cudaMemcpyAsync(device.overlap_adjoint.get() + late_element, &nan, sizeof(nan),
                             cudaMemcpyHostToDevice, stream));
  CHECK(run_force(device, host, stream, tiles) == 0);
  std::uint32_t system_error = 0u;
  std::uint32_t device_error = 0u;
  CUDA_CHECK(device.gradients.copy_to(tiled.data(), tiled.size(), stream));
  CUDA_CHECK(device.system_errors.copy_to(&system_error, 1u, stream));
  CUDA_CHECK(device.device_error.copy_to(&device_error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(system_error == static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kNonfiniteAdjoint));
  CHECK(device_error == static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kNonfiniteAdjoint));
  CHECK(tiled == host.seed);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_multitile_hostile_primitive_offset_is_peer_isolated() {
  HostCase host;
  std::string error;
  CHECK(make_case(2u, host, error, 65u));
  constexpr std::int64_t kTiles = 2;
  constexpr std::int64_t kThreadsPerLinearTile = 64;
  for (std::size_t system = 0; system < 2u; ++system) {
    CHECK(host.integrals.matrix_offsets[system + 1u] - host.integrals.matrix_offsets[system] >
          kThreadsPerLinearTile);
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));

  CUDA_CHECK(device.upload_dynamic(host, stream));
  CHECK(run_force(device, host, stream, kTiles) == 0);
  std::vector<double> clean(host.seed.size());
  CUDA_CHECK(device.gradients.copy_to(clean.data(), clean.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  /* Corrupt the large member's first primitive boundary. Before the topology
   * and numerical phases were separated, the CTA that owned shell zero could
   * reject while another tile consumed this negative system-wide bound. */
  CUDA_CHECK(device.upload_dynamic(host, stream));
  const std::int64_t hostile_primitive_begin = -1024;
  CUDA_CHECK(cudaMemcpyAsync(device.shell_primitive_offsets.get(), &hostile_primitive_begin,
                             sizeof(hostile_primitive_begin), cudaMemcpyHostToDevice, stream));
  CHECK(run_force(device, host, stream, kTiles) == 0);

  std::vector<double> actual(host.seed.size());
  std::vector<std::uint32_t> system_errors(2u);
  std::uint32_t device_error = 0u;
  CUDA_CHECK(device.gradients.copy_to(actual.data(), actual.size(), stream));
  CUDA_CHECK(device.system_errors.copy_to(system_errors.data(), system_errors.size(), stream));
  CUDA_CHECK(device.device_error.copy_to(&device_error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(system_errors[0] ==
        static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kInvalidShellMetadata));
  CHECK(system_errors[1] == static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kSuccess));
  CHECK(device_error == static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kInvalidShellMetadata));

  const std::size_t failed_end = static_cast<std::size_t>(host.basis.atom_offsets[1] * 3);
  CHECK(std::equal(actual.begin(), actual.begin() + static_cast<std::ptrdiff_t>(failed_end),
                   host.seed.begin()));
  /* The healthy large peer still runs concurrent atomic force reductions.
   * Reject cross-system contamination while allowing repeat-order roundoff. */
  CHECK(std::equal(
      actual.begin() + static_cast<std::ptrdiff_t>(failed_end), actual.end(),
      clean.begin() + static_cast<std::ptrdiff_t>(failed_end),
      [](double actual_value, double clean_value) { return near(actual_value, clean_value); }));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_cpu_parity_seed_and_ragged_batches(); status != 0) {
    return status;
  }
  if (const int status = test_component_finite_differences(); status != 0) {
    return status;
  }
  if (const int status = test_requested_and_failed_poison_peers(); status != 0) {
    return status;
  }
  if (const int status = test_changed_input_graph_replay(); status != 0) {
    return status;
  }
  if (const int status = test_large_singleton_tiling_graph_and_late_failure(); status != 0) {
    return status;
  }
  return test_multitile_hostile_primitive_offset_is_peer_isolated();
}
