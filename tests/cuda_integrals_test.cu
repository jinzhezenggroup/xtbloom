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

#include "backends/cuda/gfn2_integral_tasks.hpp"
#include "backends/cuda/gfn2_integrals.cuh"
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

using xtbloom::detail::cuda::Gfn2H0DevicePlan;
using xtbloom::detail::cuda::Gfn2IntegralDeviceBatch;
using xtbloom::detail::cuda::Gfn2IntegralDeviceError;
using xtbloom::detail::cuda::Gfn2IntegralDeviceWorkspace;
using xtbloom::detail::cuda::Gfn2IntegralHostTaskDomains;
using xtbloom::detail::cuda::Gfn2IntegralLinearLaunchShape;
using xtbloom::detail::cuda::Gfn2IntegralShellPairTask;
using xtbloom::detail::cuda::kGfn2IntegralLinearBlockBudget;
using xtbloom::detail::cuda::make_gfn2_integral_linear_launch_shape;
using xtbloom::detail::gfn2::BasisPlan;
using xtbloom::detail::gfn2::H0Plan;
using xtbloom::detail::gfn2::IntegralPlan;

constexpr std::uint64_t kPlanToken = 0x62a5d31ec774b809ULL;
constexpr double kSentinel = -917.25;

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

  cudaError_t copy_to(T* destination, std::size_t count, cudaStream_t stream = nullptr) const {
    if (destination == nullptr || count > count_) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
  }

  cudaError_t fill(const T& value, std::size_t count, cudaStream_t stream = nullptr) {
    if (count > count_) {
      return cudaErrorInvalidValue;
    }
    std::vector<T> host(count, value);
    return copy_from(host.data(), count, stream);
  }

  T* get() { return data_; }
  const T* get() const { return data_; }
  std::size_t size() const { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

template <typename T>
cudaError_t upload(DeviceBuffer<T>& target, const std::vector<T>& source, cudaStream_t stream) {
  cudaError_t status = target.allocate(source.size());
  return status == cudaSuccess ? target.copy_from(source.data(), source.size(), stream) : status;
}

bool near(double actual, double expected, double absolute = 2.0e-13, double relative = 2.0e-13) {
  return std::abs(actual - expected) <=
         absolute + relative * std::max(std::abs(actual), std::abs(expected));
}

int count_kernel_grid(cudaGraph_t graph, dim3 expected_grid, std::size_t& matches,
                      dim3 expected_block = dim3{}, bool match_block = false) {
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
    const dim3 block = parameters.blockDim;
    const bool block_matches =
        !match_block ||
        (block.x == expected_block.x && block.y == expected_block.y && block.z == expected_block.z);
    if (grid.x == expected_grid.x && grid.y == expected_grid.y && grid.z == expected_grid.z &&
        block_matches) {
      ++matches;
    }
  }
  return 0;
}

struct HostCase {
  BasisPlan basis;
  IntegralPlan integrals;
  H0Plan h0;
  Gfn2IntegralHostTaskDomains tasks;
  std::vector<std::int64_t> shell_pair_offsets;
  std::int64_t maximum_system_shells = 0;
  std::vector<double> positions;
  std::vector<double> coordination;
  std::vector<double> overlap;
  std::vector<double> dipole;
  std::vector<double> quadrupole;
  std::vector<double> hamiltonian;
};

void append_system(std::size_t system, std::size_t batch_size,
                   std::vector<std::int32_t>& atomic_numbers, std::vector<double>& positions) {
  const double origin = 4.0 * static_cast<double>(system);
  auto append_atom = [&](std::int32_t number, double x, double y, double z) {
    atomic_numbers.push_back(number);
    positions.push_back(origin + x);
    positions.push_back(y);
    positions.push_back(z);
  };
  if (batch_size == 1u || system == 0u) {
    append_atom(14, 0.0, 0.0, 0.0);
    append_atom(14, 1.1, -0.7, 2.3);
  } else if (system == 1u) {
    /* A noncoincident short-range member exercises the small-R recurrence. */
    append_atom(1, 0.0, 0.0, 0.0);
    append_atom(1, 1.0e-5, -2.0e-5, 3.0e-5);
  } else if (system == 2u) {
    append_atom(8, 0.13, -0.21, 0.34);
    append_atom(1, 1.47, -0.76, 2.08);
  } else if (system == 3u) {
    /* Atom permutation plus a 90-degree z rotation of the preceding motif. */
    append_atom(1, 0.76, 1.47, 2.08);
    append_atom(8, 0.21, 0.13, 0.34);
  } else if (system == 4u) {
    /* Transition-metal peers make every ss/sp/sd/pp/pd/dd class observable
     * in forward, H0, and strict cross-atom force task accounting. */
    append_atom(26, -0.4, 0.2, -0.1);
    append_atom(26, 1.6, -0.5, 0.9);
  } else if (system % 7u == 0u) {
    /* Deliberately empty ragged members. */
  } else if (system % 3u == 0u) {
    append_atom(6, -0.2, 0.4, -0.6);
    append_atom(1, 1.1, -0.3, 0.8);
  } else {
    append_atom(system % 2u == 0u ? 8 : 1, 0.02 * static_cast<double>(system % 5u),
                -0.03 * static_cast<double>(system % 11u),
                0.01 * static_cast<double>(system % 13u));
  }
}

bool make_case(std::size_t batch_size, HostCase& data, std::string& error) {
  std::vector<std::int64_t> atom_offsets(batch_size + 1u, 0);
  std::vector<std::int32_t> atomic_numbers;
  for (std::size_t system = 0; system < batch_size; ++system) {
    atom_offsets[system] = static_cast<std::int64_t>(atomic_numbers.size());
    append_system(system, batch_size, atomic_numbers, data.positions);
  }
  atom_offsets[batch_size] = static_cast<std::int64_t>(atomic_numbers.size());
  if (xtbloom::detail::gfn2::make_basis_plan(static_cast<std::int64_t>(batch_size),
                                             static_cast<std::int64_t>(atomic_numbers.size()),
                                             atom_offsets.data(), atomic_numbers.data(), data.basis,
                                             error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_integral_plan(data.basis, data.integrals, error) !=
          XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_h0_plan(data.basis, data.integrals, atomic_numbers.data(),
                                          data.h0, error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  data.shell_pair_offsets.resize(batch_size + 1u, 0);
  for (std::size_t system = 0; system < batch_size; ++system) {
    const std::int64_t shells =
        data.basis.batch_shell_offsets[system + 1u] - data.basis.batch_shell_offsets[system];
    data.maximum_system_shells = std::max(data.maximum_system_shells, shells);
    data.shell_pair_offsets[system + 1u] = data.shell_pair_offsets[system] + shells * shells;
  }
  if (data.shell_pair_offsets != data.h0.shell_pair_offsets) {
    error = "test shell-pair offsets disagree with H0Plan";
    return false;
  }
  if (xtbloom::detail::cuda::make_gfn2_integral_task_domains(
          data.basis, data.shell_pair_offsets, data.tasks, error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  const std::size_t atoms = static_cast<std::size_t>(data.basis.total_atoms);
  const std::size_t matrices = static_cast<std::size_t>(data.integrals.total_matrix_elements);
  data.coordination.resize(atoms);
  for (std::size_t atom = 0; atom < atoms; ++atom) {
    data.coordination[atom] = 0.35 + 0.07 * static_cast<double>((atom * 11u) % 19u);
  }
  data.overlap.resize(matrices);
  data.dipole.resize(3u * matrices);
  data.quadrupole.resize(6u * matrices);
  data.hamiltonian.resize(matrices);
  std::vector<double> cpu_workspace((data.integrals.workspace_size_bytes + sizeof(double) - 1u) /
                                    sizeof(double));
  return xtbloom::detail::gfn2::evaluate_overlap_cpu(
             data.basis, data.integrals, data.positions.data(), data.overlap.data(),
             cpu_workspace.data(), cpu_workspace.size() * sizeof(double),
             error) == XTBLOOM_STATUS_SUCCESS &&
         xtbloom::detail::gfn2::evaluate_multipole_cpu(
             data.basis, data.integrals, data.positions.data(), data.dipole.data(),
             data.quadrupole.data(), cpu_workspace.data(), cpu_workspace.size() * sizeof(double),
             error) == XTBLOOM_STATUS_SUCCESS &&
         xtbloom::detail::gfn2::evaluate_h0_cpu(
             data.basis, data.integrals, data.h0, data.positions.data(), data.coordination.data(),
             data.overlap.data(), data.hamiltonian.data(), error) == XTBLOOM_STATUS_SUCCESS;
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
  DeviceBuffer<Gfn2IntegralShellPairTask> forward_generic_tasks;
  DeviceBuffer<Gfn2IntegralShellPairTask> forward_ss_tasks;
  DeviceBuffer<Gfn2IntegralShellPairTask> h0_generic_tasks;
  DeviceBuffer<Gfn2IntegralShellPairTask> h0_ss_tasks;
  DeviceBuffer<Gfn2IntegralShellPairTask> force_generic_tasks;
  DeviceBuffer<Gfn2IntegralShellPairTask> force_ss_tasks;
  DeviceBuffer<double> atomic_radii;
  DeviceBuffer<double> shell_levels;
  DeviceBuffer<double> shell_coordination_scale;
  DeviceBuffer<double> shell_polynomial;
  DeviceBuffer<double> shell_pair_scale;
  DeviceBuffer<double> positions;
  DeviceBuffer<double> coordination;
  DeviceBuffer<double> overlap;
  DeviceBuffer<double> dipole;
  DeviceBuffer<double> quadrupole;
  DeviceBuffer<double> hamiltonian;
  DeviceBuffer<double> overlap_scratch;
  DeviceBuffer<double> dipole_scratch;
  DeviceBuffer<double> quadrupole_scratch;
  DeviceBuffer<double> h0_scratch;
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
    UPLOAD(forward_generic_tasks, host.tasks.forward_generic)
    UPLOAD(forward_ss_tasks, host.tasks.forward_ss)
    UPLOAD(h0_generic_tasks, host.tasks.h0_generic)
    UPLOAD(h0_ss_tasks, host.tasks.h0_ss)
    UPLOAD(force_generic_tasks, host.tasks.force_generic)
    UPLOAD(force_ss_tasks, host.tasks.force_ss)
    UPLOAD(atomic_radii, host.h0.atomic_radii)
    UPLOAD(shell_levels, host.h0.shell_levels)
    UPLOAD(shell_coordination_scale, host.h0.shell_coordination_scale)
    UPLOAD(shell_polynomial, host.h0.shell_polynomial)
    UPLOAD(shell_pair_scale, host.h0.shell_pair_scale)
    UPLOAD(positions, host.positions)
    UPLOAD(coordination, host.coordination)
#undef UPLOAD
    const std::size_t matrices = static_cast<std::size_t>(host.integrals.total_matrix_elements);
    const std::size_t systems = static_cast<std::size_t>(host.basis.batch_size);
    if (status == cudaSuccess) {
      status = overlap.allocate(matrices);
    }
    if (status == cudaSuccess) {
      status = dipole.allocate(3u * matrices);
    }
    if (status == cudaSuccess) {
      status = quadrupole.allocate(6u * matrices);
    }
    if (status == cudaSuccess) {
      status = hamiltonian.allocate(matrices);
    }
    if (status == cudaSuccess) {
      status = overlap_scratch.allocate(matrices);
    }
    if (status == cudaSuccess) {
      status = dipole_scratch.allocate(3u * matrices);
    }
    if (status == cudaSuccess) {
      status = quadrupole_scratch.allocate(6u * matrices);
    }
    if (status == cudaSuccess) {
      status = h0_scratch.allocate(matrices);
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
    return status;
  }

  Gfn2IntegralDeviceBatch batch(const HostCase& host, bool compact = false) const {
    Gfn2IntegralDeviceBatch value{
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
    if (compact) {
      value.use_compact_tasks = 1u;
      value.forward_generic_task_count =
          static_cast<std::int64_t>(host.tasks.forward_generic.size());
      value.forward_ss_task_count = static_cast<std::int64_t>(host.tasks.forward_ss.size());
      value.h0_generic_task_count = static_cast<std::int64_t>(host.tasks.h0_generic.size());
      value.h0_ss_task_count = static_cast<std::int64_t>(host.tasks.h0_ss.size());
      value.force_generic_task_count = static_cast<std::int64_t>(host.tasks.force_generic.size());
      value.force_ss_task_count = static_cast<std::int64_t>(host.tasks.force_ss.size());
      value.forward_generic_tasks =
          host.tasks.forward_generic.empty() ? nullptr : forward_generic_tasks.get();
      value.forward_ss_tasks = host.tasks.forward_ss.empty() ? nullptr : forward_ss_tasks.get();
      value.h0_generic_tasks = host.tasks.h0_generic.empty() ? nullptr : h0_generic_tasks.get();
      value.h0_ss_tasks = host.tasks.h0_ss.empty() ? nullptr : h0_ss_tasks.get();
      value.force_generic_tasks =
          host.tasks.force_generic.empty() ? nullptr : force_generic_tasks.get();
      value.force_ss_tasks = host.tasks.force_ss.empty() ? nullptr : force_ss_tasks.get();
    }
    return value;
  }

  Gfn2H0DevicePlan h0_plan(const HostCase& host) const {
    return Gfn2H0DevicePlan{host.basis.total_atoms,
                            host.basis.total_shells,
                            host.basis.total_shells,
                            host.basis.total_shells,
                            host.shell_pair_offsets.back(),
                            kPlanToken,
                            atomic_radii.get(),
                            shell_levels.get(),
                            shell_coordination_scale.get(),
                            shell_polynomial.get(),
                            shell_pair_scale.get()};
  }

  Gfn2IntegralDeviceWorkspace workspace(const HostCase& host) {
    const std::int64_t matrices = host.integrals.total_matrix_elements;
    return Gfn2IntegralDeviceWorkspace{overlap_scratch.get(),
                                       matrices,
                                       dipole_scratch.get(),
                                       3 * matrices,
                                       quadrupole_scratch.get(),
                                       6 * matrices,
                                       h0_scratch.get(),
                                       matrices,
                                       sequence_active.get(),
                                       1,
                                       kPlanToken};
  }

  cudaError_t seed_outputs(const HostCase& host, cudaStream_t stream) {
    const std::size_t matrices = static_cast<std::size_t>(host.integrals.total_matrix_elements);
    cudaError_t status = overlap.fill(kSentinel, matrices, stream);
    if (status == cudaSuccess) {
      status = dipole.fill(kSentinel, 3u * matrices, stream);
    }
    if (status == cudaSuccess) {
      status = quadrupole.fill(kSentinel, 6u * matrices, stream);
    }
    return status == cudaSuccess ? hamiltonian.fill(kSentinel, matrices, stream) : status;
  }
};

int run_forward(DeviceFixture& device, const HostCase& host, cudaStream_t stream,
                std::int64_t linear_tiles_per_system = 1, bool compact = false) {
  Gfn2IntegralDeviceBatch batch = device.batch(host, compact);
  batch.linear_tiles_per_system = linear_tiles_per_system;
  const Gfn2IntegralDeviceWorkspace workspace = device.workspace(host);
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_integral_device_errors_cuda(
      batch.batch_size, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_integrals_cuda(
      batch, device.positions.get(), device.overlap.get(), device.dipole.get(),
      device.quadrupole.get(), workspace, device.system_errors.get(), device.device_error.get(),
      stream));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_integral_device_errors_cuda(
      batch.batch_size, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_h0_cuda(
      batch, device.h0_plan(host), device.positions.get(), device.coordination.get(),
      device.overlap.get(), device.hamiltonian.get(), workspace, device.system_errors.get(),
      device.device_error.get(), stream));
  return 0;
}

int check_results(DeviceFixture& device, const HostCase& host, cudaStream_t stream) {
  std::vector<double> overlap(host.overlap.size());
  std::vector<double> dipole(host.dipole.size());
  std::vector<double> quadrupole(host.quadrupole.size());
  std::vector<double> hamiltonian(host.hamiltonian.size());
  std::vector<std::uint32_t> errors(static_cast<std::size_t>(host.basis.batch_size));
  std::uint32_t diagnostic = 0u;
  CUDA_CHECK(device.overlap.copy_to(overlap.data(), overlap.size(), stream));
  CUDA_CHECK(device.dipole.copy_to(dipole.data(), dipole.size(), stream));
  CUDA_CHECK(device.quadrupole.copy_to(quadrupole.data(), quadrupole.size(), stream));
  CUDA_CHECK(device.hamiltonian.copy_to(hamiltonian.data(), hamiltonian.size(), stream));
  CUDA_CHECK(device.system_errors.copy_to(errors.data(), errors.size(), stream));
  CUDA_CHECK(device.device_error.copy_to(&diagnostic, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(std::all_of(errors.begin(), errors.end(), [](std::uint32_t value) { return value == 0u; }));
  CHECK(diagnostic == 0u);
  for (std::size_t index = 0; index < overlap.size(); ++index) {
    CHECK(near(overlap[index], host.overlap[index]));
    CHECK(near(hamiltonian[index], host.hamiltonian[index], 3.0e-13, 3.0e-13));
  }
  for (std::size_t index = 0; index < dipole.size(); ++index) {
    CHECK(near(dipole[index], host.dipole[index], 3.0e-13, 3.0e-13));
  }
  for (std::size_t index = 0; index < quadrupole.size(); ++index) {
    CHECK(near(quadrupole[index], host.quadrupole[index], 8.0e-13, 5.0e-13));
  }
  return 0;
}

int test_compact_forward_bitwise_parity() {
  HostCase host;
  std::string error;
  CHECK(make_case(8u, host, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));

  Gfn2IntegralHostTaskDomains expected;
  const auto shell_class = [](std::uint8_t first, std::uint8_t second) {
    const std::uint8_t lower = std::min(first, second);
    const std::uint8_t upper = std::max(first, second);
    if (lower == 0u) return static_cast<std::size_t>(upper);
    return lower == 1u ? static_cast<std::size_t>(upper + 2u) : 5u;
  };
  const auto primitive_signature =
      [](std::vector<xtbloom::detail::cuda::Gfn2IntegralPrimitiveSignatureAccounting>& signatures,
         std::uint8_t bra_l, std::uint8_t ket_l, std::int64_t bra_primitives,
         std::int64_t ket_primitives)
      -> xtbloom::detail::cuda::Gfn2IntegralPrimitiveSignatureAccounting& {
    const auto existing =
        std::find_if(signatures.begin(), signatures.end(), [&](const auto& signature) {
          return signature.bra_angular_momentum == bra_l &&
                 signature.ket_angular_momentum == ket_l &&
                 signature.bra_primitives == bra_primitives &&
                 signature.ket_primitives == ket_primitives;
        });
    if (existing != signatures.end()) return *existing;
    signatures.push_back({bra_l, ket_l, bra_primitives, ket_primitives, 0, 0, 0, 0u, 0u, 0u});
    return signatures.back();
  };
  for (std::int64_t system = 0; system < host.basis.batch_size; ++system) {
    const std::int64_t shell_begin =
        host.basis.batch_shell_offsets[static_cast<std::size_t>(system)];
    const std::int64_t shell_end =
        host.basis.batch_shell_offsets[static_cast<std::size_t>(system + 1)];
    const std::int64_t shells = shell_end - shell_begin;
    for (std::int64_t local_pair = 0; local_pair < shells * shells; ++local_pair) {
      const std::int64_t bra_shell = shell_begin + local_pair / shells;
      const std::int64_t ket_shell = shell_begin + local_pair % shells;
      const std::int64_t bra_atom = host.basis.shell_to_atom[static_cast<std::size_t>(bra_shell)];
      const std::int64_t ket_atom = host.basis.shell_to_atom[static_cast<std::size_t>(ket_shell)];
      const std::uint8_t bra_l = host.basis.angular_momenta[static_cast<std::size_t>(bra_shell)];
      const std::uint8_t ket_l = host.basis.angular_momenta[static_cast<std::size_t>(ket_shell)];
      const std::size_t klass = shell_class(bra_l, ket_l);
      const auto bra_primitives =
          host.basis.shell_primitive_offsets[static_cast<std::size_t>(bra_shell + 1)] -
          host.basis.shell_primitive_offsets[static_cast<std::size_t>(bra_shell)];
      const auto ket_primitives =
          host.basis.shell_primitive_offsets[static_cast<std::size_t>(ket_shell + 1)] -
          host.basis.shell_primitive_offsets[static_cast<std::size_t>(ket_shell)];
      const auto primitive_work = static_cast<std::uint64_t>(bra_primitives * ket_primitives);
      const Gfn2IntegralShellPairTask task{
          static_cast<std::uint32_t>(system), static_cast<std::uint32_t>(local_pair),
          static_cast<std::uint32_t>(bra_shell), static_cast<std::uint32_t>(ket_shell)};
      (bra_l == 0u && ket_l == 0u ? expected.h0_ss : expected.h0_generic).push_back(task);
      ++expected.accounting.h0_shell_classes[klass];
      expected.accounting.h0_primitive_work += primitive_work;
      auto& signature = primitive_signature(expected.accounting.primitive_signatures, bra_l, ket_l,
                                            bra_primitives, ket_primitives);
      ++signature.h0_tasks;
      signature.h0_primitive_work += primitive_work;
      if (bra_atom < ket_atom || (bra_atom == ket_atom && bra_shell <= ket_shell)) {
        (bra_l == 0u && ket_l == 0u ? expected.forward_ss : expected.forward_generic)
            .push_back(task);
        ++expected.accounting.forward_shell_classes[klass];
        expected.accounting.forward_primitive_work += primitive_work;
        ++signature.forward_tasks;
        signature.forward_primitive_work += primitive_work;
      }
      if (bra_atom < ket_atom) {
        (bra_l == 0u && ket_l == 0u ? expected.force_ss : expected.force_generic).push_back(task);
        ++expected.accounting.force_shell_classes[klass];
        expected.accounting.force_primitive_work += primitive_work;
        ++signature.force_tasks;
        signature.force_primitive_work += primitive_work;
      }
    }
  }
  const auto same_tasks = [](const std::vector<Gfn2IntegralShellPairTask>& actual,
                             const std::vector<Gfn2IntegralShellPairTask>& reference) {
    return actual.size() == reference.size() &&
           std::equal(actual.begin(), actual.end(), reference.begin(),
                      [](const auto& first, const auto& second) {
                        return first.system == second.system &&
                               first.local_pair == second.local_pair &&
                               first.bra_shell == second.bra_shell &&
                               first.ket_shell == second.ket_shell;
                      });
  };
  CHECK(same_tasks(host.tasks.forward_generic, expected.forward_generic));
  CHECK(same_tasks(host.tasks.forward_ss, expected.forward_ss));
  CHECK(same_tasks(host.tasks.h0_generic, expected.h0_generic));
  CHECK(same_tasks(host.tasks.h0_ss, expected.h0_ss));
  CHECK(same_tasks(host.tasks.force_generic, expected.force_generic));
  CHECK(same_tasks(host.tasks.force_ss, expected.force_ss));
  const auto& accounting = host.tasks.accounting;
  const auto forward_tasks =
      static_cast<std::int64_t>(host.tasks.forward_generic.size() + host.tasks.forward_ss.size());
  const auto force_tasks =
      static_cast<std::int64_t>(host.tasks.force_generic.size() + host.tasks.force_ss.size());
  CHECK(accounting.capacity_slots ==
        host.basis.batch_size * host.maximum_system_shells * host.maximum_system_shells);
  CHECK(accounting.dense_live_pairs == host.shell_pair_offsets.back());
  CHECK(accounting.capacity_tail_slots == accounting.capacity_slots - accounting.dense_live_pairs);
  CHECK(accounting.forward_symmetry_exits == accounting.dense_live_pairs - forward_tasks);
  CHECK(accounting.force_ineligibility_exits == accounting.dense_live_pairs - force_tasks);
  CHECK(std::accumulate(accounting.forward_shell_classes.begin(),
                        accounting.forward_shell_classes.end(), std::int64_t{0}) == forward_tasks);
  CHECK(std::accumulate(accounting.h0_shell_classes.begin(), accounting.h0_shell_classes.end(),
                        std::int64_t{0}) == accounting.dense_live_pairs);
  CHECK(std::accumulate(accounting.force_shell_classes.begin(),
                        accounting.force_shell_classes.end(), std::int64_t{0}) == force_tasks);
  CHECK(accounting.forward_shell_classes == expected.accounting.forward_shell_classes);
  CHECK(accounting.h0_shell_classes == expected.accounting.h0_shell_classes);
  CHECK(accounting.force_shell_classes == expected.accounting.force_shell_classes);
  CHECK(std::all_of(accounting.forward_shell_classes.begin(),
                    accounting.forward_shell_classes.end(),
                    [](std::int64_t count) { return count > 0; }));
  CHECK(std::all_of(accounting.h0_shell_classes.begin(), accounting.h0_shell_classes.end(),
                    [](std::int64_t count) { return count > 0; }));
  CHECK(std::all_of(accounting.force_shell_classes.begin(), accounting.force_shell_classes.end(),
                    [](std::int64_t count) { return count > 0; }));
  CHECK(accounting.forward_primitive_work == expected.accounting.forward_primitive_work);
  CHECK(accounting.h0_primitive_work == expected.accounting.h0_primitive_work);
  CHECK(accounting.force_primitive_work == expected.accounting.force_primitive_work);
  CHECK(accounting.primitive_signatures.size() == expected.accounting.primitive_signatures.size());
  CHECK(std::equal(
      accounting.primitive_signatures.begin(), accounting.primitive_signatures.end(),
      expected.accounting.primitive_signatures.begin(), [](const auto& actual, const auto& wanted) {
        return actual.bra_angular_momentum == wanted.bra_angular_momentum &&
               actual.ket_angular_momentum == wanted.ket_angular_momentum &&
               actual.bra_primitives == wanted.bra_primitives &&
               actual.ket_primitives == wanted.ket_primitives &&
               actual.forward_tasks == wanted.forward_tasks && actual.h0_tasks == wanted.h0_tasks &&
               actual.force_tasks == wanted.force_tasks &&
               actual.forward_primitive_work == wanted.forward_primitive_work &&
               actual.h0_primitive_work == wanted.h0_primitive_work &&
               actual.force_primitive_work == wanted.force_primitive_work;
      }));
  xtbloom::detail::cuda::Gfn2IntegralTaskAccounting selection_accounting{};
  selection_accounting.dense_live_pairs =
      xtbloom::detail::cuda::kGfn2IntegralCompactMinimumDensePairs - 1;
  CHECK(!xtbloom::detail::cuda::prefer_gfn2_compact_integral_tasks(selection_accounting));
  selection_accounting.dense_live_pairs =
      xtbloom::detail::cuda::kGfn2IntegralCompactMinimumDensePairs;
  CHECK(xtbloom::detail::cuda::prefer_gfn2_compact_integral_tasks(selection_accounting));

  auto download = [&](std::vector<double>& overlap, std::vector<double>& dipole,
                      std::vector<double>& quadrupole, std::vector<double>& hamiltonian) -> int {
    overlap.resize(host.overlap.size());
    dipole.resize(host.dipole.size());
    quadrupole.resize(host.quadrupole.size());
    hamiltonian.resize(host.hamiltonian.size());
    CUDA_CHECK(device.overlap.copy_to(overlap.data(), overlap.size(), stream));
    CUDA_CHECK(device.dipole.copy_to(dipole.data(), dipole.size(), stream));
    CUDA_CHECK(device.quadrupole.copy_to(quadrupole.data(), quadrupole.size(), stream));
    CUDA_CHECK(device.hamiltonian.copy_to(hamiltonian.data(), hamiltonian.size(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
  };

  CUDA_CHECK(device.seed_outputs(host, stream));
  CHECK(run_forward(device, host, stream, 1, false) == 0);
  std::vector<double> legacy_overlap;
  std::vector<double> legacy_dipole;
  std::vector<double> legacy_quadrupole;
  std::vector<double> legacy_hamiltonian;
  CHECK(download(legacy_overlap, legacy_dipole, legacy_quadrupole, legacy_hamiltonian) == 0);

  CUDA_CHECK(device.seed_outputs(host, stream));
  CHECK(run_forward(device, host, stream, 1, true) == 0);
  std::vector<double> compact_overlap;
  std::vector<double> compact_dipole;
  std::vector<double> compact_quadrupole;
  std::vector<double> compact_hamiltonian;
  CHECK(download(compact_overlap, compact_dipole, compact_quadrupole, compact_hamiltonian) == 0);
  CHECK(compact_overlap == legacy_overlap);
  CHECK(compact_dipole == legacy_dipole);
  CHECK(std::equal(
      compact_quadrupole.begin(), compact_quadrupole.end(), legacy_quadrupole.begin(),
      [](double compact, double legacy) { return near(compact, legacy, 2.0e-13, 2.0e-13); }));
  CHECK(compact_hamiltonian == legacy_hamiltonian);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_cpu_parity_and_ragged_batches() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (std::size_t batch_size : {1u, 8u, 32u, 128u}) {
    HostCase host;
    std::string error;
    CHECK(make_case(batch_size, host, error));
    DeviceFixture device;
    CUDA_CHECK(device.initialize(host, stream));
    CUDA_CHECK(device.seed_outputs(host, stream));
    CHECK(run_forward(device, host, stream) == 0);
    CHECK(check_results(device, host, stream) == 0);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_cuda_graph_replay() {
  HostCase host;
  std::string error;
  CHECK(make_case(8u, host, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CUDA_CHECK(device.seed_outputs(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const Gfn2IntegralDeviceBatch batch = device.batch(host, true);
  const Gfn2IntegralDeviceWorkspace workspace = device.workspace(host);
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_integral_device_errors_cuda(
      batch.batch_size, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_integrals_cuda(
      batch, device.positions.get(), device.overlap.get(), device.dipole.get(),
      device.quadrupole.get(), workspace, device.system_errors.get(), device.device_error.get(),
      stream));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_integral_device_errors_cuda(
      batch.batch_size, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_h0_cuda(
      batch, device.h0_plan(host), device.positions.get(), device.coordination.get(),
      device.overlap.get(), device.hamiltonian.get(), workspace, device.system_errors.get(),
      device.device_error.get(), stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  std::size_t compact_nodes = 0u;
  const auto ss_grid = static_cast<unsigned int>((host.tasks.forward_ss.size() + 63u) / 64u);
  CHECK(count_kernel_grid(graph, dim3(ss_grid, 1u, 1u), compact_nodes) == 0);
  CHECK(compact_nodes >= 1u);
  compact_nodes = 0u;
  CHECK(count_kernel_grid(graph,
                          dim3(static_cast<unsigned int>(host.tasks.h0_generic.size()), 1u, 1u),
                          compact_nodes, dim3(16u, 1u, 1u), true) == 0);
  CHECK(compact_nodes >= 1u);
  compact_nodes = 0u;
  const auto h0_ss_grid = static_cast<unsigned int>((host.tasks.h0_ss.size() + 63u) / 64u);
  CHECK(count_kernel_grid(graph, dim3(h0_ss_grid, 1u, 1u), compact_nodes) == 0);
  CHECK(compact_nodes >= 1u);
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CHECK(check_results(device, host, stream) == 0);
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

bool slice_is(const std::vector<double>& values, std::size_t begin, std::size_t end,
              double expected) {
  return std::all_of(values.begin() + static_cast<std::ptrdiff_t>(begin),
                     values.begin() + static_cast<std::ptrdiff_t>(end),
                     [expected](double value) { return value == expected; });
}

int test_peer_failure_isolation() {
  HostCase host;
  std::string error;
  CHECK(make_case(8u, host, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CUDA_CHECK(device.seed_outputs(host, stream));
  constexpr std::size_t failed_system = 2u;
  const std::size_t failed_atom = static_cast<std::size_t>(host.basis.atom_offsets[failed_system]);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(cudaMemcpyAsync(device.positions.get() + failed_atom * 3u, &nan, sizeof(nan),
                             cudaMemcpyHostToDevice, stream));
  const Gfn2IntegralDeviceBatch batch = device.batch(host, true);
  const Gfn2IntegralDeviceWorkspace workspace = device.workspace(host);
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_integral_device_errors_cuda(
      batch.batch_size, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_integrals_cuda(
      batch, device.positions.get(), device.overlap.get(), device.dipole.get(),
      device.quadrupole.get(), workspace, device.system_errors.get(), device.device_error.get(),
      stream));
  std::vector<double> overlap(host.overlap.size());
  std::vector<std::uint32_t> errors(8u);
  std::uint32_t diagnostic = 0u;
  CUDA_CHECK(device.overlap.copy_to(overlap.data(), overlap.size(), stream));
  CUDA_CHECK(device.system_errors.copy_to(errors.data(), errors.size(), stream));
  CUDA_CHECK(device.device_error.copy_to(&diagnostic, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(errors[failed_system] ==
        static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kNonfinitePosition));
  CHECK(diagnostic == static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kNonfinitePosition));
  for (std::size_t system = 0; system < 8u; ++system) {
    const std::size_t begin = static_cast<std::size_t>(host.integrals.matrix_offsets[system]);
    const std::size_t end = static_cast<std::size_t>(host.integrals.matrix_offsets[system + 1u]);
    if (system == failed_system) {
      CHECK(slice_is(overlap, begin, end, kSentinel));
    } else {
      CHECK(errors[system] == 0u);
      for (std::size_t element = begin; element < end; ++element) {
        CHECK(near(overlap[element], host.overlap[element]));
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

enum class OffsetPartition {
  kMatrix,
  kShell,
  kOrbital,
  kPrimitive,
  kShellPair,
};

cudaError_t mutate_offset(DeviceFixture& device, OffsetPartition partition, std::int64_t value,
                          cudaStream_t stream) {
  std::int64_t* target = nullptr;
  switch (partition) {
    case OffsetPartition::kMatrix:
      target = device.matrix_offsets.get() + 1;
      break;
    case OffsetPartition::kShell:
      target = device.batch_shell_offsets.get() + 1;
      break;
    case OffsetPartition::kOrbital:
      target = device.batch_orbital_offsets.get() + 1;
      break;
    case OffsetPartition::kPrimitive:
      target = device.shell_primitive_offsets.get() + 1;
      break;
    case OffsetPartition::kShellPair:
      target = device.shell_pair_offsets.get() + 1;
      break;
  }
  return cudaMemcpyAsync(target, &value, sizeof(value), cudaMemcpyHostToDevice, stream);
}

cudaError_t restore_offsets(DeviceFixture& device, const HostCase& host, OffsetPartition partition,
                            cudaStream_t stream) {
  switch (partition) {
    case OffsetPartition::kMatrix:
      return device.matrix_offsets.copy_from(host.integrals.matrix_offsets.data(),
                                             host.integrals.matrix_offsets.size(), stream);
    case OffsetPartition::kShell:
      return device.batch_shell_offsets.copy_from(host.basis.batch_shell_offsets.data(),
                                                  host.basis.batch_shell_offsets.size(), stream);
    case OffsetPartition::kOrbital:
      return device.batch_orbital_offsets.copy_from(
          host.basis.batch_orbital_offsets.data(), host.basis.batch_orbital_offsets.size(), stream);
    case OffsetPartition::kPrimitive:
      return device.shell_primitive_offsets.copy_from(host.basis.shell_primitive_offsets.data(),
                                                      host.basis.shell_primitive_offsets.size(),
                                                      stream);
    case OffsetPartition::kShellPair:
      return device.shell_pair_offsets.copy_from(host.shell_pair_offsets.data(),
                                                 host.shell_pair_offsets.size(), stream);
  }
  return cudaErrorInvalidValue;
}

int test_extreme_offsets_and_sticky_error_fail_closed() {
  HostCase host;
  std::string error;
  CHECK(make_case(1u, host, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  const Gfn2IntegralDeviceBatch batch = device.batch(host, true);
  const Gfn2IntegralDeviceWorkspace workspace = device.workspace(host);
  const std::size_t matrices = host.overlap.size();
  std::vector<double> overlap(matrices);
  std::vector<double> dipole(3u * matrices);
  std::vector<double> quadrupole(6u * matrices);
  std::vector<double> hamiltonian(matrices);
  for (OffsetPartition partition :
       {OffsetPartition::kMatrix, OffsetPartition::kShell, OffsetPartition::kOrbital,
        OffsetPartition::kPrimitive, OffsetPartition::kShellPair}) {
    for (std::int64_t extreme :
         {std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::max()}) {
      CUDA_CHECK(device.seed_outputs(host, stream));
      CUDA_CHECK(mutate_offset(device, partition, extreme, stream));
      CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_integral_device_errors_cuda(
          1, device.system_errors.get(), device.device_error.get(), stream));
      CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_integrals_cuda(
          batch, device.positions.get(), device.overlap.get(), device.dipole.get(),
          device.quadrupole.get(), workspace, device.system_errors.get(), device.device_error.get(),
          stream));
      std::uint32_t system_error = 0u;
      CUDA_CHECK(device.overlap.copy_to(overlap.data(), overlap.size(), stream));
      CUDA_CHECK(device.dipole.copy_to(dipole.data(), dipole.size(), stream));
      CUDA_CHECK(device.quadrupole.copy_to(quadrupole.data(), quadrupole.size(), stream));
      CUDA_CHECK(device.system_errors.copy_to(&system_error, 1u, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(system_error != static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kSuccess));
      CHECK(slice_is(overlap, 0u, overlap.size(), kSentinel));
      CHECK(slice_is(dipole, 0u, dipole.size(), kSentinel));
      CHECK(slice_is(quadrupole, 0u, quadrupole.size(), kSentinel));

      /* The same corrupt partition must fail closed in the standalone H0 path. */
      CUDA_CHECK(device.overlap.copy_from(host.overlap.data(), host.overlap.size(), stream));
      CUDA_CHECK(device.hamiltonian.fill(kSentinel, matrices, stream));
      CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_integral_device_errors_cuda(
          1, device.system_errors.get(), device.device_error.get(), stream));
      CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_h0_cuda(
          batch, device.h0_plan(host), device.positions.get(), device.coordination.get(),
          device.overlap.get(), device.hamiltonian.get(), workspace, device.system_errors.get(),
          device.device_error.get(), stream));
      system_error = 0u;
      CUDA_CHECK(device.hamiltonian.copy_to(hamiltonian.data(), hamiltonian.size(), stream));
      CUDA_CHECK(device.system_errors.copy_to(&system_error, 1u, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(system_error != static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kSuccess));
      CHECK(slice_is(hamiltonian, 0u, hamiltonian.size(), kSentinel));
      CUDA_CHECK(restore_offsets(device, host, partition, stream));
    }
  }

  CUDA_CHECK(device.seed_outputs(host, stream));
  const std::uint32_t sticky = 0x5a5a5a5aU;
  const std::uint32_t zero = 0u;
  CUDA_CHECK(device.system_errors.copy_from(&zero, 1u, stream));
  CUDA_CHECK(device.device_error.copy_from(&sticky, 1u, stream));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_integrals_cuda(
      batch, device.positions.get(), device.overlap.get(), device.dipole.get(),
      device.quadrupole.get(), workspace, device.system_errors.get(), device.device_error.get(),
      stream));
  std::uint32_t diagnostic = 0u;
  CUDA_CHECK(device.overlap.copy_to(overlap.data(), overlap.size(), stream));
  CUDA_CHECK(device.dipole.copy_to(dipole.data(), dipole.size(), stream));
  CUDA_CHECK(device.quadrupole.copy_to(quadrupole.data(), quadrupole.size(), stream));
  CUDA_CHECK(device.device_error.copy_to(&diagnostic, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(diagnostic == sticky);
  CHECK(slice_is(overlap, 0u, overlap.size(), kSentinel));
  CHECK(slice_is(dipole, 0u, dipole.size(), kSentinel));
  CHECK(slice_is(quadrupole, 0u, quadrupole.size(), kSentinel));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_aliases_are_rejected_synchronously() {
  HostCase host;
  std::string error;
  CHECK(make_case(1u, host, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CUDA_CHECK(device.seed_outputs(host, stream));
  const Gfn2IntegralDeviceBatch batch = device.batch(host, true);
  const Gfn2IntegralDeviceWorkspace workspace = device.workspace(host);

  Gfn2IntegralDeviceWorkspace public_scratch_alias = workspace;
  public_scratch_alias.overlap_scratch = device.overlap.get();
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_integrals_cuda(
            batch, device.positions.get(), device.overlap.get(), device.dipole.get(),
            device.quadrupole.get(), public_scratch_alias, device.system_errors.get(),
            device.device_error.get(), stream) == cudaErrorInvalidValue);

  auto* const public_diagnostic_alias = reinterpret_cast<std::uint32_t*>(device.overlap.get());
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_integrals_cuda(
            batch, device.positions.get(), device.overlap.get(), device.dipole.get(),
            device.quadrupole.get(), workspace, device.system_errors.get(), public_diagnostic_alias,
            stream) == cudaErrorInvalidValue);

  Gfn2IntegralDeviceWorkspace h0_public_scratch_alias = workspace;
  h0_public_scratch_alias.h0_scratch = device.hamiltonian.get();
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_h0_cuda(
            batch, device.h0_plan(host), device.positions.get(), device.coordination.get(),
            device.overlap.get(), device.hamiltonian.get(), h0_public_scratch_alias,
            device.system_errors.get(), device.device_error.get(),
            stream) == cudaErrorInvalidValue);

  Gfn2IntegralDeviceBatch missing_task_pointer = batch;
  missing_task_pointer.forward_ss_tasks = nullptr;
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_integrals_cuda(
            missing_task_pointer, device.positions.get(), device.overlap.get(), device.dipole.get(),
            device.quadrupole.get(), workspace, device.system_errors.get(),
            device.device_error.get(), stream) == cudaErrorInvalidValue);

  Gfn2IntegralDeviceBatch aliased_task_pointer = batch;
  aliased_task_pointer.forward_ss_tasks =
      reinterpret_cast<const Gfn2IntegralShellPairTask*>(device.overlap.get());
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_integrals_cuda(
            aliased_task_pointer, device.positions.get(), device.overlap.get(), device.dipole.get(),
            device.quadrupole.get(), workspace, device.system_errors.get(),
            device.device_error.get(), stream) == cudaErrorInvalidValue);

  Gfn2IntegralDeviceBatch invalid_selector = batch;
  invalid_selector.use_compact_tasks = 2u;
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_integrals_cuda(
            invalid_selector, device.positions.get(), device.overlap.get(), device.dipole.get(),
            device.quadrupole.get(), workspace, device.system_errors.get(),
            device.device_error.get(), stream) == cudaErrorInvalidValue);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_corrupt_compact_task_fails_closed() {
  HostCase host;
  std::string error;
  CHECK(make_case(8u, host, error));
  CHECK(!host.tasks.forward_ss.empty());
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CUDA_CHECK(device.seed_outputs(host, stream));
  const Gfn2IntegralShellPairTask invalid{std::numeric_limits<std::uint32_t>::max(), 0u};
  CUDA_CHECK(device.forward_ss_tasks.copy_from(&invalid, 1u, stream));
  const Gfn2IntegralDeviceBatch batch = device.batch(host, true);
  const Gfn2IntegralDeviceWorkspace workspace = device.workspace(host);
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_integral_device_errors_cuda(
      batch.batch_size, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_integrals_cuda(
      batch, device.positions.get(), device.overlap.get(), device.dipole.get(),
      device.quadrupole.get(), workspace, device.system_errors.get(), device.device_error.get(),
      stream));
  std::vector<double> overlap(host.overlap.size());
  std::vector<std::uint32_t> errors(static_cast<std::size_t>(host.basis.batch_size));
  std::uint32_t diagnostic = 0u;
  CUDA_CHECK(device.overlap.copy_to(overlap.data(), overlap.size(), stream));
  CUDA_CHECK(device.system_errors.copy_to(errors.data(), errors.size(), stream));
  CUDA_CHECK(device.device_error.copy_to(&diagnostic, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(diagnostic == static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kInvalidShellMetadata));
  CHECK(std::all_of(errors.begin(), errors.end(), [](std::uint32_t code) {
    return code == static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kInvalidShellMetadata);
  }));
  CHECK(slice_is(overlap, 0u, overlap.size(), kSentinel));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_h0_failure_atomicity() {
  HostCase host;
  std::string error;
  CHECK(make_case(8u, host, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CUDA_CHECK(device.seed_outputs(host, stream));
  CHECK(run_forward(device, host, stream, 1, true) == 0);
  CUDA_CHECK(device.hamiltonian.fill(kSentinel, host.hamiltonian.size(), stream));
  constexpr std::size_t failed_system = 3u;
  const std::size_t failed_atom = static_cast<std::size_t>(host.basis.atom_offsets[failed_system]);
  const double infinity = std::numeric_limits<double>::infinity();
  CUDA_CHECK(cudaMemcpyAsync(device.coordination.get() + failed_atom, &infinity, sizeof(infinity),
                             cudaMemcpyHostToDevice, stream));
  const Gfn2IntegralDeviceBatch batch = device.batch(host, true);
  const Gfn2IntegralDeviceWorkspace workspace = device.workspace(host);
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_integral_device_errors_cuda(
      batch.batch_size, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_h0_cuda(
      batch, device.h0_plan(host), device.positions.get(), device.coordination.get(),
      device.overlap.get(), device.hamiltonian.get(), workspace, device.system_errors.get(),
      device.device_error.get(), stream));
  std::vector<double> hamiltonian(host.hamiltonian.size());
  std::vector<std::uint32_t> errors(8u);
  CUDA_CHECK(device.hamiltonian.copy_to(hamiltonian.data(), hamiltonian.size(), stream));
  CUDA_CHECK(device.system_errors.copy_to(errors.data(), errors.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(errors[failed_system] ==
        static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kInvalidCoordination));
  for (std::size_t system = 0; system < 8u; ++system) {
    const std::size_t begin = static_cast<std::size_t>(host.integrals.matrix_offsets[system]);
    const std::size_t end = static_cast<std::size_t>(host.integrals.matrix_offsets[system + 1u]);
    if (system == failed_system) {
      CHECK(slice_is(hamiltonian, begin, end, kSentinel));
    } else {
      for (std::size_t element = begin; element < end; ++element) {
        CHECK(near(hamiltonian[element], host.hamiltonian[element], 3.0e-13, 3.0e-13));
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_linear_launch_shape_tiling_and_late_failure() {
  Gfn2IntegralLinearLaunchShape shape{17u, 19u};
  CHECK(make_gfn2_integral_linear_launch_shape(3, 2, shape));
  CHECK(shape.systems == 3u && shape.tiles == 2u);
  for (const std::array<std::int64_t, 2> invalid :
       {std::array<std::int64_t, 2>{0, 1}, std::array<std::int64_t, 2>{1, 0},
        std::array<std::int64_t, 2>{1, kGfn2IntegralLinearBlockBudget + 1}}) {
    shape = {17u, 19u};
    CHECK(!make_gfn2_integral_linear_launch_shape(invalid[0], invalid[1], shape));
    CHECK(shape.systems == 17u && shape.tiles == 19u);
  }

  HostCase host;
  std::string error;
  CHECK(make_case(1u, host, error));
  CHECK(host.integrals.total_matrix_elements > 64);
  constexpr std::int64_t kTiles = 2;
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));

  auto download_outputs = [&](std::vector<double>& overlap, std::vector<double>& dipole,
                              std::vector<double>& quadrupole,
                              std::vector<double>& hamiltonian) -> int {
    overlap.resize(host.overlap.size());
    dipole.resize(host.dipole.size());
    quadrupole.resize(host.quadrupole.size());
    hamiltonian.resize(host.hamiltonian.size());
    CUDA_CHECK(device.overlap.copy_to(overlap.data(), overlap.size(), stream));
    CUDA_CHECK(device.dipole.copy_to(dipole.data(), dipole.size(), stream));
    CUDA_CHECK(device.quadrupole.copy_to(quadrupole.data(), quadrupole.size(), stream));
    CUDA_CHECK(device.hamiltonian.copy_to(hamiltonian.data(), hamiltonian.size(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
  };

  CUDA_CHECK(device.seed_outputs(host, stream));
  CHECK(run_forward(device, host, stream, 1) == 0);
  std::vector<double> single_overlap;
  std::vector<double> single_dipole;
  std::vector<double> single_quadrupole;
  std::vector<double> single_hamiltonian;
  CHECK(download_outputs(single_overlap, single_dipole, single_quadrupole, single_hamiltonian) ==
        0);

  CUDA_CHECK(device.seed_outputs(host, stream));
  CHECK(run_forward(device, host, stream, kTiles) == 0);
  std::vector<double> tiled_overlap;
  std::vector<double> tiled_dipole;
  std::vector<double> tiled_quadrupole;
  std::vector<double> tiled_hamiltonian;
  CHECK(download_outputs(tiled_overlap, tiled_dipole, tiled_quadrupole, tiled_hamiltonian) == 0);
  CHECK(tiled_overlap == single_overlap);
  CHECK(tiled_dipole == single_dipole);
  CHECK(tiled_quadrupole == single_quadrupole);
  CHECK(tiled_hamiltonian == single_hamiltonian);

  CUDA_CHECK(device.seed_outputs(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  Gfn2IntegralDeviceBatch tiled_batch = device.batch(host);
  tiled_batch.linear_tiles_per_system = kTiles;
  const Gfn2IntegralDeviceWorkspace workspace = device.workspace(host);
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_integral_device_errors_cuda(
      tiled_batch.batch_size, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_integrals_cuda(
      tiled_batch, device.positions.get(), device.overlap.get(), device.dipole.get(),
      device.quadrupole.get(), workspace, device.system_errors.get(), device.device_error.get(),
      stream));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_integral_device_errors_cuda(
      tiled_batch.batch_size, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_h0_cuda(
      tiled_batch, device.h0_plan(host), device.positions.get(), device.coordination.get(),
      device.overlap.get(), device.hamiltonian.get(), workspace, device.system_errors.get(),
      device.device_error.get(), stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  std::size_t tiled_nodes = 0u;
  CHECK(count_kernel_grid(graph, dim3(1u, static_cast<unsigned int>(kTiles), 1u), tiled_nodes) ==
        0);
  /* Integral publication plus H0 preflight and H0 publication use this fixed grid. */
  CHECK(tiled_nodes == 3u);
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CHECK(download_outputs(tiled_overlap, tiled_dipole, tiled_quadrupole, tiled_hamiltonian) == 0);
  CHECK(tiled_overlap == single_overlap);
  CHECK(tiled_dipole == single_dipole);
  CHECK(tiled_quadrupole == single_quadrupole);
  CHECK(tiled_hamiltonian == single_hamiltonian);
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));

  Gfn2IntegralDeviceBatch invalid_batch = device.batch(host);
  invalid_batch.linear_tiles_per_system = 0;
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_integrals_cuda(
            invalid_batch, device.positions.get(), device.overlap.get(), device.dipole.get(),
            device.quadrupole.get(), workspace, device.system_errors.get(),
            device.device_error.get(), stream) == cudaErrorInvalidValue);
  invalid_batch.linear_tiles_per_system = kGfn2IntegralLinearBlockBudget + 1;
  CHECK(xtbloom::detail::cuda::evaluate_gfn2_integrals_cuda(
            invalid_batch, device.positions.get(), device.overlap.get(), device.dipole.get(),
            device.quadrupole.get(), workspace, device.system_errors.get(),
            device.device_error.get(), stream) == cudaErrorInvalidValue);

  /* Element 64 is owned by the second tile. Its nonfinite value must suppress
   * the complete H0 publication rather than leak an earlier tile's scratch. */
  CUDA_CHECK(device.overlap.copy_from(host.overlap.data(), host.overlap.size(), stream));
  CUDA_CHECK(device.hamiltonian.fill(kSentinel, host.hamiltonian.size(), stream));
  const double nan = std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(cudaMemcpyAsync(device.overlap.get() + 64, &nan, sizeof(nan), cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(xtbloom::detail::cuda::reset_gfn2_integral_device_errors_cuda(
      tiled_batch.batch_size, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::evaluate_gfn2_h0_cuda(
      tiled_batch, device.h0_plan(host), device.positions.get(), device.coordination.get(),
      device.overlap.get(), device.hamiltonian.get(), workspace, device.system_errors.get(),
      device.device_error.get(), stream));
  std::vector<double> failed_h0(host.hamiltonian.size());
  std::uint32_t system_error = 0u;
  CUDA_CHECK(device.hamiltonian.copy_to(failed_h0.data(), failed_h0.size(), stream));
  CUDA_CHECK(device.system_errors.copy_to(&system_error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(system_error == static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kNonfiniteOverlap));
  CHECK(slice_is(failed_h0, 0u, failed_h0.size(), kSentinel));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_cpu_parity_and_ragged_batches(); status != 0) {
    return status;
  }
  if (const int status = test_cuda_graph_replay(); status != 0) {
    return status;
  }
  if (const int status = test_compact_forward_bitwise_parity(); status != 0) {
    return status;
  }
  if (const int status = test_peer_failure_isolation(); status != 0) {
    return status;
  }
  if (const int status = test_extreme_offsets_and_sticky_error_fail_closed(); status != 0) {
    return status;
  }
  if (const int status = test_aliases_are_rejected_synchronously(); status != 0) {
    return status;
  }
  if (const int status = test_corrupt_compact_task_fails_closed(); status != 0) {
    return status;
  }
  if (const int status = test_h0_failure_atomicity(); status != 0) {
    return status;
  }
  return test_linear_launch_shape_tiling_and_late_failure();
}
