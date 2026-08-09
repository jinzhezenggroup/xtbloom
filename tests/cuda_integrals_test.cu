#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

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

struct HostCase {
  BasisPlan basis;
  IntegralPlan integrals;
  H0Plan h0;
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

int run_forward(DeviceFixture& device, const HostCase& host, cudaStream_t stream) {
  const Gfn2IntegralDeviceBatch batch = device.batch(host);
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
  const Gfn2IntegralDeviceBatch batch = device.batch(host);
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
  const Gfn2IntegralDeviceBatch batch = device.batch(host);
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
  const Gfn2IntegralDeviceBatch batch = device.batch(host);
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
  const Gfn2IntegralDeviceBatch batch = device.batch(host);
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
  CUDA_CHECK(cudaStreamSynchronize(stream));
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
  CHECK(run_forward(device, host, stream) == 0);
  CUDA_CHECK(device.hamiltonian.fill(kSentinel, host.hamiltonian.size(), stream));
  constexpr std::size_t failed_system = 3u;
  const std::size_t failed_atom = static_cast<std::size_t>(host.basis.atom_offsets[failed_system]);
  const double infinity = std::numeric_limits<double>::infinity();
  CUDA_CHECK(cudaMemcpyAsync(device.coordination.get() + failed_atom, &infinity, sizeof(infinity),
                             cudaMemcpyHostToDevice, stream));
  const Gfn2IntegralDeviceBatch batch = device.batch(host);
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

}  // namespace

int main() {
  if (const int status = test_cpu_parity_and_ragged_batches(); status != 0) {
    return status;
  }
  if (const int status = test_cuda_graph_replay(); status != 0) {
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
  return test_h0_failure_atomicity();
}
