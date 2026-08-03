#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_external_point_charges.cuh"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/external_point_charges.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using gpuxtb::detail::cuda::Gfn2ExternalPointChargeDeviceBatch;
using gpuxtb::detail::cuda::Gfn2ExternalPointChargeDeviceError;
using gpuxtb::detail::cuda::Gfn2ExternalPointChargeForceDeviceWorkspace;
using gpuxtb::detail::cuda::Gfn2ForceDeviceActivity;
using gpuxtb::detail::gfn2::BasisPlan;
using gpuxtb::detail::gfn2::ExternalPointChargePlan;

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
    if (count == 0u) {
      return cudaSuccess;
    }
    return cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
  }

  cudaError_t copy_from(const T* source, std::size_t count, cudaStream_t stream = nullptr) {
    if (count > count_) {
      return cudaErrorInvalidValue;
    }
    if (count == 0u) {
      return cudaSuccess;
    }
    return cudaMemcpyAsync(data_, source, count * sizeof(T), cudaMemcpyHostToDevice, stream);
  }

  cudaError_t copy_to(T* destination, std::size_t count, cudaStream_t stream = nullptr) const {
    if (count > count_) {
      return cudaErrorInvalidValue;
    }
    if (count == 0u) {
      return cudaSuccess;
    }
    return cudaMemcpyAsync(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
  }

  T* get() { return data_; }
  const T* get() const { return data_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

template <typename T>
cudaError_t allocate_and_copy(DeviceBuffer<T>& buffer, const std::vector<T>& values,
                              cudaStream_t stream) {
  cudaError_t status = buffer.allocate(values.size());
  if (status != cudaSuccess) {
    return status;
  }
  return buffer.copy_from(values.data(), values.size(), stream);
}

struct DeviceFixture {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_point_charges = 0;
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> shell_offsets;
  DeviceBuffer<std::int64_t> point_offsets;
  DeviceBuffer<std::int64_t> shell_to_atom;
  DeviceBuffer<double> shell_hardness;
  DeviceBuffer<double> qm_positions;
  DeviceBuffer<double> point_positions;
  DeviceBuffer<double> point_charges;
  DeviceBuffer<double> point_hardnesses;

  cudaError_t upload(const ExternalPointChargePlan& plan, const std::vector<double>& qm_xyz,
                     const std::vector<double>& point_xyz, const std::vector<double>& point_q,
                     const std::vector<double>& point_gamma, cudaStream_t stream) {
    batch_size = plan.batch_size;
    total_atoms = plan.total_atoms;
    total_shells = plan.total_shells;
    total_point_charges = plan.total_point_charges;
    cudaError_t status = allocate_and_copy(atom_offsets, plan.atom_offsets, stream);
    if (status == cudaSuccess) {
      status = allocate_and_copy(shell_offsets, plan.batch_shell_offsets, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(point_offsets, plan.point_charge_offsets, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(shell_to_atom, plan.shell_to_atom, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(shell_hardness, plan.shell_hardness, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(qm_positions, qm_xyz, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(point_positions, point_xyz, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(point_charges, point_q, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(point_hardnesses, point_gamma, stream);
    }
    return status;
  }

  Gfn2ExternalPointChargeDeviceBatch batch() const {
    return {batch_size,
            total_atoms,
            total_shells,
            total_point_charges,
            atom_offsets.get(),
            shell_offsets.get(),
            point_offsets.get(),
            shell_to_atom.get(),
            shell_hardness.get(),
            qm_positions.get(),
            point_positions.get(),
            point_charges.get(),
            point_hardnesses.get()};
  }
};

bool near(double actual, double expected, double absolute_tolerance,
          double relative_tolerance = 0.0) {
  const double scale = std::max(std::abs(actual), std::abs(expected));
  return std::abs(actual - expected) <= absolute_tolerance + relative_tolerance * scale;
}

bool make_plan(const std::vector<std::int64_t>& atom_offsets,
               const std::vector<std::int32_t>& atomic_numbers,
               const std::vector<std::int64_t>* point_offsets, BasisPlan& basis,
               ExternalPointChargePlan& plan, std::string& error) {
  const std::int64_t batch_size = static_cast<std::int64_t>(atom_offsets.size() - 1u);
  if (gpuxtb::detail::gfn2::make_basis_plan(
          batch_size, static_cast<std::int64_t>(atomic_numbers.size()), atom_offsets.data(),
          atomic_numbers.data(), basis, error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  const std::int64_t point_count = point_offsets == nullptr ? 0 : point_offsets->back();
  return gpuxtb::detail::gfn2::make_external_point_charge_plan(
             basis, atomic_numbers.data(), point_count,
             point_offsets == nullptr ? nullptr : point_offsets->data(), plan,
             error) == GPUXTB_STATUS_SUCCESS;
}

bool evaluate_cpu_embedding_energy(const ExternalPointChargePlan& plan,
                                   const std::vector<double>& qm_positions,
                                   const std::vector<double>& point_positions,
                                   const std::vector<double>& point_charges,
                                   const std::vector<double>& point_hardnesses,
                                   const std::vector<double>& shell_charges, double& energy,
                                   std::string& error) {
  std::vector<double> potentials(static_cast<std::size_t>(plan.total_shells));
  std::vector<double> energies(static_cast<std::size_t>(plan.batch_size));
  if (gpuxtb::detail::gfn2::evaluate_external_point_charge_potential_cpu(
          plan, qm_positions.data(), point_positions.data(), point_charges.data(),
          point_hardnesses.data(), potentials.data(), error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::add_external_point_charge_energy_cpu(plan, shell_charges.data(),
                                                                 potentials.data(), energies.data(),
                                                                 error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  energy = 0.0;
  for (double value : energies) {
    energy += value;
  }
  return true;
}

int test_ragged_water_golden_stream_and_graph() {
  /* Systems: water, an H with zero PCs, empty QM with one inert PC, and C/Si. */
  const std::vector<std::int64_t> atom_offsets{0, 3, 4, 4, 6};
  const std::vector<std::int32_t> atomic_numbers{8, 1, 1, 1, 6, 14};
  const std::vector<std::int64_t> point_offsets{0, 1, 1, 2, 5};
  const std::vector<double> qm_positions{
      0.0, 0.0,  0.0, 1.43233673, 0.0, 1.10715266, -1.43233673, 0.0,  1.10715266,
      2.0, -1.0, 0.5, -0.4,       0.2, 1.1,        1.3,         -0.7, 0.5,
  };
  const std::vector<double> point_positions{
      4.0, 0.0, 0.0, 9.0, 9.0, 9.0, -0.4, 0.2, 1.1, 4.0, 0.1, -0.7, 2.5, 2.1, 1.4,
  };
  const std::vector<double> point_charges{0.5, 9.0, 0.1, -0.8, 0.35};
  const std::vector<double> point_hardnesses{0.405771, 0.7, 0.31, 999.0, 0.8};

  BasisPlan basis;
  ExternalPointChargePlan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, atomic_numbers, &point_offsets, basis, plan, error));
  std::vector<double> shell_charges(static_cast<std::size_t>(plan.total_shells));
  constexpr std::array<double, 4> water_shell_charges{
      0.26189717923223715,
      -0.8260775955268945,
      0.23530677010797196,
      0.3288736461866886,
  };
  std::copy(water_shell_charges.begin(), water_shell_charges.end(), shell_charges.begin());
  for (std::size_t shell = water_shell_charges.size(); shell < shell_charges.size(); ++shell) {
    shell_charges[shell] = 0.3 * std::sin(0.71 * static_cast<double>(shell + 1u)) - 0.11;
  }

  std::vector<double> expected_potentials(shell_charges.size(), 77.0);
  std::vector<double> expected_energies{0.25, -0.5, 1.5, 2.5};
  std::vector<double> expected_qm_forces(qm_positions.size());
  std::vector<double> expected_point_forces(point_positions.size());
  for (std::size_t coordinate = 0; coordinate < expected_qm_forces.size(); ++coordinate) {
    expected_qm_forces[coordinate] = 0.01 * static_cast<double>(coordinate) - 0.08;
  }
  for (std::size_t coordinate = 0; coordinate < expected_point_forces.size(); ++coordinate) {
    expected_point_forces[coordinate] = 0.005 * static_cast<double>(coordinate) - 0.03;
  }
  const std::vector<double> initial_energies = expected_energies;
  const std::vector<double> initial_qm_forces = expected_qm_forces;
  const std::vector<double> initial_point_forces = expected_point_forces;
  CHECK(gpuxtb::detail::gfn2::evaluate_external_point_charge_potential_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), expected_potentials.data(), error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_energy_cpu(
            plan, shell_charges.data(), expected_potentials.data(), expected_energies.data(),
            error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), shell_charges.data(), expected_qm_forces.data(),
            expected_point_forces.data(), error) == GPUXTB_STATUS_SUCCESS);

  constexpr std::array<double, 4> water_potential{
      0.10798911399580015,
      0.10997182482659495,
      0.13414824825075822,
      0.082411861188258814,
  };
  for (std::size_t shell = 0; shell < water_potential.size(); ++shell) {
    CHECK(near(expected_potentials[shell], water_potential[shell], 3.0e-16));
  }
  CHECK(near(expected_energies[0] - initial_energies[0], -0.003894135995627615, 4.0e-16));

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture fixture;
  CUDA_CHECK(
      fixture.upload(plan, qm_positions, point_positions, point_charges, point_hardnesses, stream));
  DeviceBuffer<double> device_shell_charges;
  DeviceBuffer<double> device_potentials;
  DeviceBuffer<double> device_energies;
  DeviceBuffer<double> device_qm_forces;
  DeviceBuffer<double> device_point_forces;
  DeviceBuffer<std::uint32_t> device_error;
  CUDA_CHECK(allocate_and_copy(device_shell_charges, shell_charges, stream));
  std::vector<double> actual_potentials(shell_charges.size(), 77.0);
  std::vector<double> actual_energies = initial_energies;
  std::vector<double> actual_qm_forces = initial_qm_forces;
  std::vector<double> actual_point_forces = initial_point_forces;
  CUDA_CHECK(allocate_and_copy(device_potentials, actual_potentials, stream));
  CUDA_CHECK(allocate_and_copy(device_energies, actual_energies, stream));
  CUDA_CHECK(allocate_and_copy(device_qm_forces, actual_qm_forces, stream));
  CUDA_CHECK(allocate_and_copy(device_point_forces, actual_point_forces, stream));
  CUDA_CHECK(device_error.allocate(1));

  const Gfn2ExternalPointChargeDeviceBatch batch = fixture.batch();
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_external_point_charge_device_error_cuda(
      device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_external_point_charge_potential_cuda(
      batch, device_potentials.get(), device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_energy_cuda(
      batch, device_shell_charges.get(), device_potentials.get(), device_energies.get(),
      device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_forces_cuda(
      batch, device_shell_charges.get(), device_qm_forces.get(), device_point_forces.get(),
      device_error.get(), stream));
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(device_potentials.copy_to(actual_potentials.data(), actual_potentials.size(), stream));
  CUDA_CHECK(device_energies.copy_to(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(device_qm_forces.copy_to(actual_qm_forces.data(), actual_qm_forces.size(), stream));
  CUDA_CHECK(
      device_point_forces.copy_to(actual_point_forces.data(), actual_point_forces.size(), stream));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2ExternalPointChargeDeviceError::kSuccess));
  for (std::size_t shell = 0; shell < actual_potentials.size(); ++shell) {
    CHECK(near(actual_potentials[shell], expected_potentials[shell], 2.0e-15));
  }
  for (std::size_t system = 0; system < actual_energies.size(); ++system) {
    CHECK(near(actual_energies[system], expected_energies[system], 3.0e-15));
  }
  for (std::size_t coordinate = 0; coordinate < actual_qm_forces.size(); ++coordinate) {
    CHECK(near(actual_qm_forces[coordinate], expected_qm_forces[coordinate], 8.0e-15));
  }
  for (std::size_t coordinate = 0; coordinate < actual_point_forces.size(); ++coordinate) {
    CHECK(near(actual_point_forces[coordinate], expected_point_forces[coordinate], 8.0e-15));
  }

  /* The second system has QM shells but no PCs: potential overwrites to zero. */
  for (std::int64_t shell = plan.batch_shell_offsets[1]; shell < plan.batch_shell_offsets[2];
       ++shell) {
    CHECK(actual_potentials[static_cast<std::size_t>(shell)] == 0.0);
  }
  /* The empty-QM system's inert point force remains exactly at its seed. */
  for (std::size_t axis = 0; axis < 3u; ++axis) {
    CHECK(actual_point_forces[3u + axis] == initial_point_forces[3u + axis]);
  }
  CHECK(near(actual_point_forces[0] - initial_point_forces[0], -0.0024674453618602722, 8.0e-16));
  CHECK(near(actual_point_forces[2] - initial_point_forces[2], -0.0033308918783967671, 8.0e-16));

  /* Both optional force-output modes must match independent CPU references. */
  std::vector<double> expected_qm_only(qm_positions.size());
  std::vector<double> expected_point_only(point_positions.size());
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), shell_charges.data(), expected_qm_only.data(), nullptr,
            error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), shell_charges.data(), nullptr, expected_point_only.data(),
            error) == GPUXTB_STATUS_SUCCESS);
  std::vector<double> actual_qm_only(qm_positions.size());
  CUDA_CHECK(device_qm_forces.copy_from(actual_qm_only.data(), actual_qm_only.size(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_external_point_charge_device_error_cuda(
      device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_forces_cuda(
      batch, device_shell_charges.get(), device_qm_forces.get(), nullptr, device_error.get(),
      stream));
  CUDA_CHECK(device_qm_forces.copy_to(actual_qm_only.data(), actual_qm_only.size(), stream));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == 0u);
  for (std::size_t coordinate = 0; coordinate < actual_qm_only.size(); ++coordinate) {
    CHECK(near(actual_qm_only[coordinate], expected_qm_only[coordinate], 8.0e-15));
  }

  std::vector<double> actual_point_only(point_positions.size());
  CUDA_CHECK(
      device_point_forces.copy_from(actual_point_only.data(), actual_point_only.size(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_external_point_charge_device_error_cuda(
      device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_forces_cuda(
      batch, device_shell_charges.get(), nullptr, device_point_forces.get(), device_error.get(),
      stream));
  CUDA_CHECK(
      device_point_forces.copy_to(actual_point_only.data(), actual_point_only.size(), stream));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == 0u);
  for (std::size_t coordinate = 0; coordinate < actual_point_only.size(); ++coordinate) {
    CHECK(near(actual_point_only[coordinate], expected_point_only[coordinate], 8.0e-15));
  }

  /* Successful stream capture proves the hot launchers contain no forbidden sync/allocation. */
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_external_point_charge_device_error_cuda(
      device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_external_point_charge_potential_cuda(
      batch, device_potentials.get(), device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_energy_cuda(
      batch, device_shell_charges.get(), device_potentials.get(), device_energies.get(),
      device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_forces_cuda(
      batch, device_shell_charges.get(), device_qm_forces.get(), device_point_forces.get(),
      device_error.get(), stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, 0));
  CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == 0u);
  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_gated_transactional_force_reduction() {
  /*
   * Four terminal SCC members: one healthy requested member, one unrequested
   * member, one failed member, and one requested member with a numerical
   * force failure. Terminal SUCCESS members must be force-eligible even though
   * they would be inactive for another SCC iteration.
   */
  constexpr std::uint64_t kPlanToken = 0x66a11ce5ULL;
  const std::vector<std::int64_t> atom_offsets{0, 1, 2, 3, 4};
  const std::vector<std::int32_t> atomic_numbers{1, 8, 6, 14};
  const std::vector<std::int64_t> point_offsets{0, 2, 2, 4, 6};
  const std::vector<double> qm_positions{
      0.0, 0.0, 0.0, 1.1, -0.4, 0.7, -1.3, 0.8, -0.2, 2.0, -1.1, 0.5,
  };
  const std::vector<double> point_positions{
      0.0, 0.0, 0.0, 2.4, 0.3, -0.6, -2.2, 1.1, 0.9, 0.4, 2.3, -1.7, 2.0, -1.1, 0.5, 4.2, 0.6, 1.4,
  };
  const std::vector<double> point_charges{0.9, -0.35, -0.2, 0.7, 1.2, -0.55};
  const std::vector<double> point_hardnesses{0.405771, 0.8, 1.1, 0.35, 999.0, 0.5};

  BasisPlan basis;
  ExternalPointChargePlan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, atomic_numbers, &point_offsets, basis, plan, error));
  std::vector<double> shell_charges(static_cast<std::size_t>(plan.total_shells));
  for (std::size_t shell = 0; shell < shell_charges.size(); ++shell) {
    shell_charges[shell] = 0.19 * std::sin(0.47 * static_cast<double>(shell + 1u)) - 0.08;
  }
  std::vector<double> qm_seed(qm_positions.size());
  std::vector<double> point_seed(point_positions.size());
  for (std::size_t coordinate = 0; coordinate < qm_seed.size(); ++coordinate) {
    qm_seed[coordinate] = 0.002 * static_cast<double>(coordinate + 1u) - 0.013;
  }
  for (std::size_t coordinate = 0; coordinate < point_seed.size(); ++coordinate) {
    point_seed[coordinate] = -0.0015 * static_cast<double>(coordinate + 1u) + 0.017;
  }
  std::vector<double> expected_qm = qm_seed;
  std::vector<double> expected_point = point_seed;
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), shell_charges.data(), expected_qm.data(),
            expected_point.data(), error) == GPUXTB_STATUS_SUCCESS);

  /* The shared CPU energy primitive independently gates both sides by FD. */
  constexpr double kStep = 1.0e-5;
  std::vector<double> displaced_qm = qm_positions;
  double right = 0.0;
  double left = 0.0;
  displaced_qm[0] += kStep;
  CHECK(evaluate_cpu_embedding_energy(plan, displaced_qm, point_positions, point_charges,
                                      point_hardnesses, shell_charges, right, error));
  displaced_qm[0] -= 2.0 * kStep;
  CHECK(evaluate_cpu_embedding_energy(plan, displaced_qm, point_positions, point_charges,
                                      point_hardnesses, shell_charges, left, error));
  const double numerical_qm_force = -(right - left) / (2.0 * kStep);
  if (!near(numerical_qm_force, expected_qm[0] - qm_seed[0], 1.0e-9)) {
    std::cerr << "gated QM FD mismatch: numerical=" << numerical_qm_force
              << " analytic=" << expected_qm[0] - qm_seed[0] << '\n';
    return __LINE__;
  }
  std::vector<double> displaced_points = point_positions;
  displaced_points[3] += kStep;
  CHECK(evaluate_cpu_embedding_energy(plan, qm_positions, displaced_points, point_charges,
                                      point_hardnesses, shell_charges, right, error));
  displaced_points[3] -= 2.0 * kStep;
  CHECK(evaluate_cpu_embedding_energy(plan, qm_positions, displaced_points, point_charges,
                                      point_hardnesses, shell_charges, left, error));
  const double numerical_point_force = -(right - left) / (2.0 * kStep);
  if (!near(numerical_point_force, expected_point[3] - point_seed[3], 1.0e-9)) {
    std::cerr << "gated point FD mismatch: numerical=" << numerical_point_force
              << " analytic=" << expected_point[3] - point_seed[3] << '\n';
    return __LINE__;
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture fixture;
  CUDA_CHECK(
      fixture.upload(plan, qm_positions, point_positions, point_charges, point_hardnesses, stream));
  Gfn2ExternalPointChargeDeviceBatch batch = fixture.batch();
  batch.plan_token = kPlanToken;

  DeviceBuffer<double> device_shell_charges;
  DeviceBuffer<double> device_qm_forces;
  DeviceBuffer<double> device_point_forces;
  DeviceBuffer<double> device_qm_scratch;
  DeviceBuffer<double> device_point_scratch;
  DeviceBuffer<std::uint8_t> device_requested;
  DeviceBuffer<gpuxtb_status_t> device_statuses;
  DeviceBuffer<std::uint32_t> device_system_errors;
  DeviceBuffer<std::uint32_t> device_error;
  DeviceBuffer<std::uint32_t> device_sequence;
  CUDA_CHECK(allocate_and_copy(device_shell_charges, shell_charges, stream));
  CUDA_CHECK(allocate_and_copy(device_qm_forces, qm_seed, stream));
  CUDA_CHECK(allocate_and_copy(device_point_forces, point_seed, stream));
  std::vector<double> scratch_qm(qm_positions.size(), 91.0);
  std::vector<double> scratch_point(point_positions.size(), 93.0);
  CUDA_CHECK(allocate_and_copy(device_qm_scratch, scratch_qm, stream));
  CUDA_CHECK(allocate_and_copy(device_point_scratch, scratch_point, stream));
  const std::vector<std::uint8_t> requested{1u, 0u, 1u, 1u};
  const std::vector<gpuxtb_status_t> statuses{
      GPUXTB_STATUS_SUCCESS,
      static_cast<gpuxtb_status_t>(0x12345678),
      GPUXTB_STATUS_EIGENSOLVER_FAILED,
      GPUXTB_STATUS_SUCCESS,
  };
  CUDA_CHECK(allocate_and_copy(device_requested, requested, stream));
  CUDA_CHECK(allocate_and_copy(device_statuses, statuses, stream));
  CUDA_CHECK(device_system_errors.allocate(requested.size()));
  CUDA_CHECK(device_error.allocate(1u));
  CUDA_CHECK(device_sequence.allocate(1u));

  std::vector<double> poisoned_qm = qm_positions;
  std::vector<double> poisoned_point_positions = point_positions;
  std::vector<double> poisoned_point_charges = point_charges;
  std::vector<double> poisoned_point_hardnesses = point_hardnesses;
  std::vector<double> poisoned_shell_charges = shell_charges;
  poisoned_qm[3] = std::numeric_limits<double>::quiet_NaN();
  poisoned_point_hardnesses[2] = 0.0;
  for (std::int64_t shell = plan.batch_shell_offsets[1]; shell < plan.batch_shell_offsets[2];
       ++shell) {
    poisoned_shell_charges[static_cast<std::size_t>(shell)] =
        std::numeric_limits<double>::infinity();
  }
  poisoned_qm[6] = std::numeric_limits<double>::quiet_NaN();
  poisoned_point_charges[2] = std::numeric_limits<double>::quiet_NaN();
  poisoned_qm[9] = std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(fixture.qm_positions.copy_from(poisoned_qm.data(), poisoned_qm.size(), stream));
  CUDA_CHECK(fixture.point_positions.copy_from(poisoned_point_positions.data(),
                                               poisoned_point_positions.size(), stream));
  CUDA_CHECK(fixture.point_charges.copy_from(poisoned_point_charges.data(),
                                             poisoned_point_charges.size(), stream));
  CUDA_CHECK(fixture.point_hardnesses.copy_from(poisoned_point_hardnesses.data(),
                                                poisoned_point_hardnesses.size(), stream));
  CUDA_CHECK(device_shell_charges.copy_from(poisoned_shell_charges.data(),
                                            poisoned_shell_charges.size(), stream));

  const Gfn2ForceDeviceActivity activity{device_requested.get(), device_statuses.get(),
                                         plan.batch_size, kPlanToken};
  const Gfn2ExternalPointChargeForceDeviceWorkspace workspace{
      device_qm_scratch.get(),
      static_cast<std::int64_t>(qm_positions.size()),
      device_point_scratch.get(),
      static_cast<std::int64_t>(point_positions.size()),
      device_sequence.get(),
      1,
      kPlanToken,
  };
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_external_point_charge_force_errors_cuda(
      plan.batch_size, device_system_errors.get(), device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_gated_forces_cuda(
      batch, activity, device_shell_charges.get(), device_qm_forces.get(),
      device_point_forces.get(), workspace, device_system_errors.get(), device_error.get(),
      stream));

  std::vector<double> actual_qm(qm_seed.size());
  std::vector<double> actual_point(point_seed.size());
  std::vector<std::uint32_t> system_errors(requested.size());
  std::uint32_t semantic_error = 0u;
  CUDA_CHECK(device_qm_forces.copy_to(actual_qm.data(), actual_qm.size(), stream));
  CUDA_CHECK(device_point_forces.copy_to(actual_point.data(), actual_point.size(), stream));
  CUDA_CHECK(device_qm_scratch.copy_to(scratch_qm.data(), scratch_qm.size(), stream));
  CUDA_CHECK(device_point_scratch.copy_to(scratch_point.data(), scratch_point.size(), stream));
  CUDA_CHECK(device_system_errors.copy_to(system_errors.data(), system_errors.size(), stream));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(system_errors[0] == 0u && system_errors[1] == 0u && system_errors[2] == 0u);
  CHECK(system_errors[3] ==
        static_cast<std::uint32_t>(Gfn2ExternalPointChargeDeviceError::kNonfiniteQmPosition));
  CHECK(semantic_error == system_errors[3]);
  for (std::size_t system = 0; system < requested.size(); ++system) {
    const bool published = system == 0u;
    for (std::int64_t atom = atom_offsets[system]; atom < atom_offsets[system + 1u]; ++atom) {
      for (std::size_t axis = 0; axis < 3u; ++axis) {
        const std::size_t coordinate = static_cast<std::size_t>(atom) * 3u + axis;
        CHECK(near(actual_qm[coordinate], published ? expected_qm[coordinate] : qm_seed[coordinate],
                   8.0e-15));
      }
    }
    for (std::int64_t point = point_offsets[system]; point < point_offsets[system + 1u]; ++point) {
      for (std::size_t axis = 0; axis < 3u; ++axis) {
        const std::size_t coordinate = static_cast<std::size_t>(point) * 3u + axis;
        CHECK(near(actual_point[coordinate],
                   published ? expected_point[coordinate] : point_seed[coordinate], 8.0e-15));
      }
    }
  }
  /* The coincident point in the healthy member contributes exactly zero. */
  CHECK(actual_point[0] == point_seed[0] && actual_point[1] == point_seed[1] &&
        actual_point[2] == point_seed[2]);
  for (std::size_t axis = 0; axis < 3u; ++axis) {
    double total_increment = actual_qm[axis] - qm_seed[axis];
    for (std::int64_t point = point_offsets[0]; point < point_offsets[1]; ++point) {
      const std::size_t coordinate = static_cast<std::size_t>(point) * 3u + axis;
      total_increment += actual_point[coordinate] - point_seed[coordinate];
    }
    CHECK(near(total_increment, 0.0, 2.0e-17));
  }
  for (std::size_t coordinate = 3u; coordinate < 6u; ++coordinate) {
    CHECK(scratch_qm[coordinate] == 91.0);
  }
  for (std::size_t coordinate = 6u; coordinate < 12u; ++coordinate) {
    CHECK(scratch_point[coordinate] == 93.0);
  }

  /* A pre-existing sequence error suppresses every numerical read and write. */
  CUDA_CHECK(device_qm_forces.copy_from(qm_seed.data(), qm_seed.size(), stream));
  CUDA_CHECK(device_point_forces.copy_from(point_seed.data(), point_seed.size(), stream));
  const std::vector<std::uint32_t> clear_errors(requested.size(), 0u);
  const std::uint32_t incoming_error =
      static_cast<std::uint32_t>(Gfn2ExternalPointChargeDeviceError::kCacheMismatch);
  CUDA_CHECK(device_system_errors.copy_from(clear_errors.data(), clear_errors.size(), stream));
  CUDA_CHECK(device_error.copy_from(&incoming_error, 1u, stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_gated_forces_cuda(
      batch, activity, device_shell_charges.get(), device_qm_forces.get(),
      device_point_forces.get(), workspace, device_system_errors.get(), device_error.get(),
      stream));
  CUDA_CHECK(device_qm_forces.copy_to(actual_qm.data(), actual_qm.size(), stream));
  CUDA_CHECK(device_point_forces.copy_to(actual_point.data(), actual_point.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(actual_qm == qm_seed && actual_point == point_seed);

  /* Capture/replay the post-SCC gate with every terminal member successful. */
  const std::vector<std::uint8_t> all_requested(requested.size(), 1u);
  const std::vector<gpuxtb_status_t> all_success(requested.size(), GPUXTB_STATUS_SUCCESS);
  CUDA_CHECK(device_requested.copy_from(all_requested.data(), all_requested.size(), stream));
  CUDA_CHECK(device_statuses.copy_from(all_success.data(), all_success.size(), stream));
  CUDA_CHECK(fixture.qm_positions.copy_from(qm_positions.data(), qm_positions.size(), stream));
  CUDA_CHECK(
      fixture.point_positions.copy_from(point_positions.data(), point_positions.size(), stream));
  CUDA_CHECK(fixture.point_charges.copy_from(point_charges.data(), point_charges.size(), stream));
  CUDA_CHECK(
      fixture.point_hardnesses.copy_from(point_hardnesses.data(), point_hardnesses.size(), stream));
  CUDA_CHECK(device_shell_charges.copy_from(shell_charges.data(), shell_charges.size(), stream));
  CUDA_CHECK(device_qm_forces.copy_from(qm_seed.data(), qm_seed.size(), stream));
  CUDA_CHECK(device_point_forces.copy_from(point_seed.data(), point_seed.size(), stream));
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_external_point_charge_force_errors_cuda(
      plan.batch_size, device_system_errors.get(), device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_gated_forces_cuda(
      batch, activity, device_shell_charges.get(), device_qm_forces.get(),
      device_point_forces.get(), workspace, device_system_errors.get(), device_error.get(),
      stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0u));
  CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
  CUDA_CHECK(device_qm_forces.copy_to(actual_qm.data(), actual_qm.size(), stream));
  CUDA_CHECK(device_point_forces.copy_to(actual_point.data(), actual_point.size(), stream));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == 0u);
  for (std::size_t coordinate = 0; coordinate < actual_qm.size(); ++coordinate) {
    CHECK(near(actual_qm[coordinate], expected_qm[coordinate], 8.0e-15));
  }
  for (std::size_t coordinate = 0; coordinate < actual_point.size(); ++coordinate) {
    CHECK(near(actual_point[coordinate], expected_point[coordinate], 8.0e-15));
  }
  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int run_gated_ragged_force_batch(std::int64_t batch_size, bool exercise_gate_and_aliases) {
  constexpr std::array<std::int32_t, 6> kElements{1, 6, 7, 8, 14, 16};
  const std::uint64_t plan_token = 0x66ba7000ULL + static_cast<std::uint64_t>(batch_size);
  std::vector<std::int64_t> atom_offsets{0};
  std::vector<std::int64_t> point_offsets{0};
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> qm_positions;
  std::vector<double> point_positions;
  std::vector<double> point_charges;
  std::vector<double> point_hardnesses;
  for (std::int64_t system = 0; system < batch_size; ++system) {
    const std::int64_t atom_count = 1 + system % 3;
    const std::int64_t point_count = system == 0 ? 1 : system % 4;
    const double origin = 6.0 * static_cast<double>(system);
    for (std::int64_t atom = 0; atom < atom_count; ++atom) {
      atomic_numbers.push_back(
          kElements[static_cast<std::size_t>((system + atom) % kElements.size())]);
      qm_positions.push_back(origin + 0.61 * static_cast<double>(atom));
      qm_positions.push_back(-0.23 * static_cast<double>(atom) +
                             0.01 * static_cast<double>(system));
      qm_positions.push_back(0.37 * static_cast<double>(atom) - 0.02 * static_cast<double>(system));
    }
    for (std::int64_t point = 0; point < point_count; ++point) {
      point_positions.push_back(origin + 1.2 + 0.47 * static_cast<double>(point));
      point_positions.push_back(-0.5 + 0.31 * static_cast<double>(point));
      point_positions.push_back(0.8 - 0.29 * static_cast<double>(point));
      point_charges.push_back((point % 2 == 0 ? 0.27 : -0.19) +
                              0.001 * static_cast<double>(system));
      point_hardnesses.push_back(0.36 + 0.07 * static_cast<double>((system + point) % 7));
    }
    atom_offsets.push_back(static_cast<std::int64_t>(atomic_numbers.size()));
    point_offsets.push_back(static_cast<std::int64_t>(point_charges.size()));
  }

  BasisPlan basis;
  ExternalPointChargePlan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, atomic_numbers, &point_offsets, basis, plan, error));
  std::vector<double> shell_charges(static_cast<std::size_t>(plan.total_shells));
  for (std::size_t shell = 0; shell < shell_charges.size(); ++shell) {
    shell_charges[shell] = 0.17 * std::sin(0.19 * static_cast<double>(shell + 1u)) - 0.06;
  }
  std::vector<double> qm_seed(qm_positions.size());
  std::vector<double> point_seed(point_positions.size());
  for (std::size_t coordinate = 0; coordinate < qm_seed.size(); ++coordinate) {
    qm_seed[coordinate] = 0.0003 * static_cast<double>(coordinate + 1u) - 0.021;
  }
  for (std::size_t coordinate = 0; coordinate < point_seed.size(); ++coordinate) {
    point_seed[coordinate] = -0.0002 * static_cast<double>(coordinate + 1u) + 0.014;
  }
  std::vector<double> expected_qm = qm_seed;
  std::vector<double> expected_point = point_seed;
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
            plan, qm_positions.data(), point_positions.data(), point_charges.data(),
            point_hardnesses.data(), shell_charges.data(), expected_qm.data(),
            expected_point.data(), error) == GPUXTB_STATUS_SUCCESS);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture fixture;
  CUDA_CHECK(
      fixture.upload(plan, qm_positions, point_positions, point_charges, point_hardnesses, stream));
  Gfn2ExternalPointChargeDeviceBatch batch = fixture.batch();
  batch.plan_token = plan_token;
  DeviceBuffer<double> device_shell_charges;
  DeviceBuffer<double> device_qm_forces;
  DeviceBuffer<double> device_point_forces;
  DeviceBuffer<double> device_qm_scratch;
  DeviceBuffer<double> device_point_scratch;
  DeviceBuffer<std::uint8_t> device_requested;
  DeviceBuffer<gpuxtb_status_t> device_statuses;
  DeviceBuffer<std::uint32_t> device_system_errors;
  DeviceBuffer<std::uint32_t> device_error;
  DeviceBuffer<std::uint32_t> device_sequence;
  CUDA_CHECK(allocate_and_copy(device_shell_charges, shell_charges, stream));
  CUDA_CHECK(allocate_and_copy(device_qm_forces, qm_seed, stream));
  CUDA_CHECK(allocate_and_copy(device_point_forces, point_seed, stream));
  CUDA_CHECK(device_qm_scratch.allocate(qm_seed.size()));
  CUDA_CHECK(device_point_scratch.allocate(point_seed.size()));
  std::vector<std::uint8_t> requested(static_cast<std::size_t>(batch_size), 1u);
  std::vector<gpuxtb_status_t> statuses(static_cast<std::size_t>(batch_size),
                                        GPUXTB_STATUS_SUCCESS);
  CUDA_CHECK(allocate_and_copy(device_requested, requested, stream));
  CUDA_CHECK(allocate_and_copy(device_statuses, statuses, stream));
  CUDA_CHECK(device_system_errors.allocate(static_cast<std::size_t>(batch_size)));
  CUDA_CHECK(device_error.allocate(1u));
  CUDA_CHECK(device_sequence.allocate(1u));
  const Gfn2ForceDeviceActivity activity{device_requested.get(), device_statuses.get(), batch_size,
                                         plan_token};
  const Gfn2ExternalPointChargeForceDeviceWorkspace workspace{
      device_qm_scratch.get(),
      static_cast<std::int64_t>(qm_seed.size()),
      device_point_scratch.get(),
      static_cast<std::int64_t>(point_seed.size()),
      device_sequence.get(),
      1,
      plan_token,
  };
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_external_point_charge_force_errors_cuda(
      batch_size, device_system_errors.get(), device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_gated_forces_cuda(
      batch, activity, device_shell_charges.get(), device_qm_forces.get(),
      device_point_forces.get(), workspace, device_system_errors.get(), device_error.get(),
      stream));
  std::vector<double> actual_qm(qm_seed.size());
  std::vector<double> actual_point(point_seed.size());
  std::vector<std::uint32_t> system_errors(static_cast<std::size_t>(batch_size));
  std::uint32_t semantic_error = 1u;
  CUDA_CHECK(device_qm_forces.copy_to(actual_qm.data(), actual_qm.size(), stream));
  CUDA_CHECK(device_point_forces.copy_to(actual_point.data(), actual_point.size(), stream));
  CUDA_CHECK(device_system_errors.copy_to(system_errors.data(), system_errors.size(), stream));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == 0u);
  CHECK(std::all_of(system_errors.begin(), system_errors.end(),
                    [](std::uint32_t value) { return value == 0u; }));
  for (std::size_t coordinate = 0; coordinate < actual_qm.size(); ++coordinate) {
    CHECK(near(actual_qm[coordinate], expected_qm[coordinate], 2.0e-13));
  }
  for (std::size_t coordinate = 0; coordinate < actual_point.size(); ++coordinate) {
    CHECK(near(actual_point[coordinate], expected_point[coordinate], 2.0e-13));
  }
  for (std::int64_t system = 0; system < batch_size; ++system) {
    for (std::size_t axis = 0; axis < 3u; ++axis) {
      double total_increment = 0.0;
      for (std::int64_t atom = atom_offsets[static_cast<std::size_t>(system)];
           atom < atom_offsets[static_cast<std::size_t>(system + 1)]; ++atom) {
        const std::size_t coordinate = static_cast<std::size_t>(atom) * 3u + axis;
        total_increment += actual_qm[coordinate] - qm_seed[coordinate];
      }
      for (std::int64_t point = point_offsets[static_cast<std::size_t>(system)];
           point < point_offsets[static_cast<std::size_t>(system + 1)]; ++point) {
        const std::size_t coordinate = static_cast<std::size_t>(point) * 3u + axis;
        total_increment += actual_point[coordinate] - point_seed[coordinate];
      }
      CHECK(near(total_increment, 0.0, 2.0e-13));
    }
  }

  if (exercise_gate_and_aliases) {
    /* An invalid request is peer-local and must not read its status or publish its slices. */
    requested[0] = 2u;
    statuses[0] = static_cast<gpuxtb_status_t>(0x76543210);
    CUDA_CHECK(device_requested.copy_from(requested.data(), requested.size(), stream));
    CUDA_CHECK(device_statuses.copy_from(statuses.data(), statuses.size(), stream));
    CUDA_CHECK(device_qm_forces.copy_from(qm_seed.data(), qm_seed.size(), stream));
    CUDA_CHECK(device_point_forces.copy_from(point_seed.data(), point_seed.size(), stream));
    CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_external_point_charge_force_errors_cuda(
        batch_size, device_system_errors.get(), device_error.get(), stream));
    CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_gated_forces_cuda(
        batch, activity, device_shell_charges.get(), device_qm_forces.get(),
        device_point_forces.get(), workspace, device_system_errors.get(), device_error.get(),
        stream));
    CUDA_CHECK(device_qm_forces.copy_to(actual_qm.data(), actual_qm.size(), stream));
    CUDA_CHECK(device_point_forces.copy_to(actual_point.data(), actual_point.size(), stream));
    CUDA_CHECK(device_system_errors.copy_to(system_errors.data(), system_errors.size(), stream));
    CUDA_CHECK(device_error.copy_to(&semantic_error, 1u, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(system_errors[0] ==
          static_cast<std::uint32_t>(Gfn2ExternalPointChargeDeviceError::kInvalidForceRequest));
    CHECK(semantic_error == system_errors[0]);
    for (std::int64_t system = 0; system < batch_size; ++system) {
      const bool published = system != 0;
      for (std::int64_t atom = atom_offsets[static_cast<std::size_t>(system)];
           atom < atom_offsets[static_cast<std::size_t>(system + 1)]; ++atom) {
        for (std::size_t axis = 0; axis < 3u; ++axis) {
          const std::size_t coordinate = static_cast<std::size_t>(atom) * 3u + axis;
          CHECK(near(actual_qm[coordinate],
                     published ? expected_qm[coordinate] : qm_seed[coordinate], 2.0e-13));
        }
      }
      for (std::int64_t point = point_offsets[static_cast<std::size_t>(system)];
           point < point_offsets[static_cast<std::size_t>(system + 1)]; ++point) {
        for (std::size_t axis = 0; axis < 3u; ++axis) {
          const std::size_t coordinate = static_cast<std::size_t>(point) * 3u + axis;
          CHECK(near(actual_point[coordinate],
                     published ? expected_point[coordinate] : point_seed[coordinate], 2.0e-13));
        }
      }
    }

    /* Every public/scratch/control destination is disjoint from all numerical and gate reads. */
    Gfn2ExternalPointChargeForceDeviceWorkspace alias_workspace = workspace;
    alias_workspace.qm_scratch = device_qm_forces.get();
    CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_gated_forces_cuda(
              batch, activity, device_shell_charges.get(), device_qm_forces.get(),
              device_point_forces.get(), alias_workspace, device_system_errors.get(),
              device_error.get(), stream) == cudaErrorInvalidValue);
    alias_workspace = workspace;
    alias_workspace.point_scratch = device_point_forces.get();
    CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_gated_forces_cuda(
              batch, activity, device_shell_charges.get(), device_qm_forces.get(),
              device_point_forces.get(), alias_workspace, device_system_errors.get(),
              device_error.get(), stream) == cudaErrorInvalidValue);
    Gfn2ForceDeviceActivity alias_activity = activity;
    alias_activity.requested_mask = reinterpret_cast<const std::uint8_t*>(device_sequence.get());
    CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_gated_forces_cuda(
              batch, alias_activity, device_shell_charges.get(), device_qm_forces.get(),
              device_point_forces.get(), workspace, device_system_errors.get(), device_error.get(),
              stream) == cudaErrorInvalidValue);
    alias_activity = activity;
    alias_activity.system_statuses =
        reinterpret_cast<const gpuxtb_status_t*>(device_system_errors.get());
    CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_gated_forces_cuda(
              batch, alias_activity, device_shell_charges.get(), device_qm_forces.get(),
              device_point_forces.get(), workspace, device_system_errors.get(), device_error.get(),
              stream) == cudaErrorInvalidValue);
    CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_gated_forces_cuda(
              batch, activity, device_shell_charges.get(), device_qm_forces.get(),
              device_point_forces.get(), workspace, device_system_errors.get(),
              device_system_errors.get(), stream) == cudaErrorInvalidValue);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_gated_ragged_force_batch_sizes_and_alias_inventory() {
  constexpr std::array<std::int64_t, 4> kBatchSizes{1, 8, 32, 128};
  for (const std::int64_t batch_size : kBatchSizes) {
    const int status = run_gated_ragged_force_batch(batch_size, batch_size == 8);
    if (status != 0) {
      std::cerr << "gated ragged force batch " << batch_size << " failed at line " << status
                << '\n';
      return status;
    }
  }
  return 0;
}

int test_zero_total_points_with_null_device_inputs() {
  const std::vector<std::int64_t> atom_offsets{0, 2};
  const std::vector<std::int32_t> atomic_numbers{1, 8};
  const std::vector<double> qm_positions{0.0, 0.0, 0.0, 1.5, -0.2, 0.8};
  const std::vector<double> empty;
  BasisPlan basis;
  ExternalPointChargePlan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, atomic_numbers, nullptr, basis, plan, error));
  std::vector<double> shell_charges(static_cast<std::size_t>(plan.total_shells), 0.125);
  std::vector<double> potentials(shell_charges.size(), 13.0);
  std::vector<double> energies{2.0};
  std::vector<double> qm_forces(qm_positions.size(), -0.75);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture fixture;
  CUDA_CHECK(fixture.upload(plan, qm_positions, empty, empty, empty, stream));
  const Gfn2ExternalPointChargeDeviceBatch batch = fixture.batch();
  CHECK(batch.point_positions == nullptr && batch.point_charges == nullptr &&
        batch.point_hardnesses == nullptr);
  DeviceBuffer<double> device_shell_charges;
  DeviceBuffer<double> device_potentials;
  DeviceBuffer<double> device_energies;
  DeviceBuffer<double> device_qm_forces;
  DeviceBuffer<std::uint32_t> device_error;
  CUDA_CHECK(allocate_and_copy(device_shell_charges, shell_charges, stream));
  CUDA_CHECK(allocate_and_copy(device_potentials, potentials, stream));
  CUDA_CHECK(allocate_and_copy(device_energies, energies, stream));
  CUDA_CHECK(allocate_and_copy(device_qm_forces, qm_forces, stream));
  CUDA_CHECK(device_error.allocate(1));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_external_point_charge_device_error_cuda(
      device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_external_point_charge_potential_cuda(
      batch, device_potentials.get(), device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_energy_cuda(
      batch, device_shell_charges.get(), device_potentials.get(), device_energies.get(),
      device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_forces_cuda(
      batch, device_shell_charges.get(), device_qm_forces.get(), nullptr, device_error.get(),
      stream));
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(device_potentials.copy_to(potentials.data(), potentials.size(), stream));
  CUDA_CHECK(device_energies.copy_to(energies.data(), energies.size(), stream));
  CUDA_CHECK(device_qm_forces.copy_to(qm_forces.data(), qm_forces.size(), stream));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == 0u);
  CHECK(
      std::all_of(potentials.begin(), potentials.end(), [](double value) { return value == 0.0; }));
  CHECK(energies[0] == 2.0);
  CHECK(
      std::all_of(qm_forces.begin(), qm_forces.end(), [](double value) { return value == -0.75; }));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_host_and_device_errors_are_sticky() {
  CHECK(gpuxtb::detail::cuda::reset_gfn2_external_point_charge_device_error_cuda(nullptr) ==
        cudaErrorInvalidValue);
  Gfn2ExternalPointChargeDeviceBatch invalid{};
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_external_point_charge_potential_cuda(
            invalid, nullptr, nullptr) == cudaErrorInvalidValue);
  CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_energy_cuda(
            invalid, nullptr, nullptr, nullptr, nullptr) == cudaErrorInvalidValue);
  CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_forces_cuda(
            invalid, nullptr, nullptr, nullptr, nullptr) == cudaErrorInvalidValue);

  const std::vector<std::int64_t> atom_offsets{0, 1};
  const std::vector<std::int32_t> atomic_numbers{1};
  const std::vector<std::int64_t> point_offsets{0, 1};
  const std::vector<double> qm_positions{0.0, 0.0, 0.0};
  const std::vector<double> point_positions{2.0, -0.5, 1.0};
  const std::vector<double> point_charges{0.4};
  const std::vector<double> point_hardnesses{0.7};
  BasisPlan basis;
  ExternalPointChargePlan plan;
  std::string error;
  CHECK(make_plan(atom_offsets, atomic_numbers, &point_offsets, basis, plan, error));

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture fixture;
  CUDA_CHECK(
      fixture.upload(plan, qm_positions, point_positions, point_charges, point_hardnesses, stream));
  Gfn2ExternalPointChargeDeviceBatch batch = fixture.batch();
  DeviceBuffer<double> device_potential;
  DeviceBuffer<double> device_shell_charge;
  DeviceBuffer<double> device_energy;
  DeviceBuffer<double> device_qm_force;
  DeviceBuffer<std::uint32_t> device_error;
  std::vector<double> potential(static_cast<std::size_t>(plan.total_shells), 8.0);
  std::vector<double> shell_charge(static_cast<std::size_t>(plan.total_shells), 0.2);
  std::vector<double> qm_force(3, 5.0);
  std::vector<double> point_force(3, -4.0);
  std::vector<double> energy{11.0};
  CUDA_CHECK(allocate_and_copy(device_potential, potential, stream));
  CUDA_CHECK(allocate_and_copy(device_shell_charge, shell_charge, stream));
  CUDA_CHECK(allocate_and_copy(device_energy, energy, stream));
  CUDA_CHECK(allocate_and_copy(device_qm_force, qm_force, stream));
  DeviceBuffer<double> device_point_force;
  CUDA_CHECK(allocate_and_copy(device_point_force, point_force, stream));
  CUDA_CHECK(device_error.allocate(1));

  /* The caller-owned energy seed must remain finite after accumulating E_pc. */
  energy[0] = std::numeric_limits<double>::max();
  std::fill(potential.begin(), potential.end(), 0.5 * std::numeric_limits<double>::max());
  std::fill(shell_charge.begin(), shell_charge.end(), 1.0);
  CUDA_CHECK(device_potential.copy_from(potential.data(), potential.size(), stream));
  CUDA_CHECK(device_shell_charge.copy_from(shell_charge.data(), shell_charge.size(), stream));
  CUDA_CHECK(device_energy.copy_from(energy.data(), energy.size(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_external_point_charge_device_error_cuda(
      device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_energy_cuda(
      batch, device_shell_charge.get(), device_potential.get(), device_energy.get(),
      device_error.get(), stream));
  std::uint32_t semantic_error = 0u;
  CUDA_CHECK(device_energy.copy_to(energy.data(), energy.size(), stream));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error ==
        static_cast<std::uint32_t>(Gfn2ExternalPointChargeDeviceError::kNonfinitePairArithmetic));
  CHECK(energy[0] == std::numeric_limits<double>::max());

  energy[0] = 11.0;
  std::fill(potential.begin(), potential.end(), 8.0);
  std::fill(shell_charge.begin(), shell_charge.end(), 0.2);
  CUDA_CHECK(device_potential.copy_from(potential.data(), potential.size(), stream));
  CUDA_CHECK(device_shell_charge.copy_from(shell_charge.data(), shell_charge.size(), stream));
  CUDA_CHECK(device_energy.copy_from(energy.data(), energy.size(), stream));

  /* Force accumulation must reject a finite seed plus a finite overflowing increment. */
  const std::vector<double> overflow_qm_position{1.0, 0.0, 0.0};
  const std::vector<double> overflow_point_position{0.0, 0.0, 0.0};
  const std::vector<double> overflow_point_charge{0.75 * std::numeric_limits<double>::max()};
  const std::vector<double> overflow_point_hardness{std::numeric_limits<double>::max()};
  std::vector<double> overflow_shell_charge(static_cast<std::size_t>(plan.total_shells), 1.0);
  std::vector<double> cpu_overflow_qm_force{std::numeric_limits<double>::max(), 0.0, 0.0};
  std::vector<double> cpu_overflow_point_force{-std::numeric_limits<double>::max(), 0.0, 0.0};
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
            plan, overflow_qm_position.data(), overflow_point_position.data(),
            overflow_point_charge.data(), overflow_point_hardness.data(),
            overflow_shell_charge.data(), cpu_overflow_qm_force.data(), nullptr,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(cpu_overflow_qm_force[0] == std::numeric_limits<double>::max());
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
            plan, overflow_qm_position.data(), overflow_point_position.data(),
            overflow_point_charge.data(), overflow_point_hardness.data(),
            overflow_shell_charge.data(), nullptr, cpu_overflow_point_force.data(),
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(cpu_overflow_point_force[0] == -std::numeric_limits<double>::max());

  CUDA_CHECK(fixture.qm_positions.copy_from(overflow_qm_position.data(),
                                            overflow_qm_position.size(), stream));
  CUDA_CHECK(fixture.point_positions.copy_from(overflow_point_position.data(),
                                               overflow_point_position.size(), stream));
  CUDA_CHECK(fixture.point_charges.copy_from(overflow_point_charge.data(), 1, stream));
  CUDA_CHECK(fixture.point_hardnesses.copy_from(overflow_point_hardness.data(), 1, stream));
  CUDA_CHECK(device_shell_charge.copy_from(overflow_shell_charge.data(),
                                           overflow_shell_charge.size(), stream));
  qm_force = {std::numeric_limits<double>::max(), 0.0, 0.0};
  CUDA_CHECK(device_qm_force.copy_from(qm_force.data(), qm_force.size(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_external_point_charge_device_error_cuda(
      device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_forces_cuda(
      batch, device_shell_charge.get(), device_qm_force.get(), nullptr, device_error.get(),
      stream));
  CUDA_CHECK(device_qm_force.copy_to(qm_force.data(), qm_force.size(), stream));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error ==
        static_cast<std::uint32_t>(Gfn2ExternalPointChargeDeviceError::kNonfinitePairArithmetic));
  CHECK(qm_force[0] == std::numeric_limits<double>::max());

  point_force = {-std::numeric_limits<double>::max(), 0.0, 0.0};
  CUDA_CHECK(device_point_force.copy_from(point_force.data(), point_force.size(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_external_point_charge_device_error_cuda(
      device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_forces_cuda(
      batch, device_shell_charge.get(), nullptr, device_point_force.get(), device_error.get(),
      stream));
  CUDA_CHECK(device_point_force.copy_to(point_force.data(), point_force.size(), stream));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error ==
        static_cast<std::uint32_t>(Gfn2ExternalPointChargeDeviceError::kNonfinitePairArithmetic));
  CHECK(point_force[0] == -std::numeric_limits<double>::max());

  CUDA_CHECK(fixture.qm_positions.copy_from(qm_positions.data(), qm_positions.size(), stream));
  CUDA_CHECK(
      fixture.point_positions.copy_from(point_positions.data(), point_positions.size(), stream));
  CUDA_CHECK(fixture.point_charges.copy_from(point_charges.data(), point_charges.size(), stream));
  CUDA_CHECK(
      fixture.point_hardnesses.copy_from(point_hardnesses.data(), point_hardnesses.size(), stream));
  CUDA_CHECK(device_shell_charge.copy_from(shell_charge.data(), shell_charge.size(), stream));
  std::fill(qm_force.begin(), qm_force.end(), 5.0);
  CUDA_CHECK(device_qm_force.copy_from(qm_force.data(), qm_force.size(), stream));

  /* An invalid potential partition must remain sticky and suppress energy. */
  const std::vector<std::int64_t> bad_point_offsets{0, 2};
  CUDA_CHECK(
      fixture.point_offsets.copy_from(bad_point_offsets.data(), bad_point_offsets.size(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_external_point_charge_device_error_cuda(
      device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_external_point_charge_potential_cuda(
      batch, device_potential.get(), device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_energy_cuda(
      batch, device_shell_charge.get(), device_potential.get(), device_energy.get(),
      device_error.get(), stream));
  CUDA_CHECK(device_energy.copy_to(energy.data(), 1, stream));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error ==
        static_cast<std::uint32_t>(Gfn2ExternalPointChargeDeviceError::kInvalidOffsets));
  CHECK(energy[0] == 11.0);

  CUDA_CHECK(fixture.point_offsets.copy_from(point_offsets.data(), point_offsets.size(), stream));
  const std::vector<std::int64_t> bad_shell_to_atom{1};
  CUDA_CHECK(fixture.shell_to_atom.copy_from(bad_shell_to_atom.data(), 1, stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_external_point_charge_device_error_cuda(
      device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_external_point_charge_potential_cuda(
      batch, device_potential.get(), device_error.get(), stream));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error ==
        static_cast<std::uint32_t>(Gfn2ExternalPointChargeDeviceError::kInvalidShellMetadata));

  CUDA_CHECK(fixture.shell_to_atom.copy_from(plan.shell_to_atom.data(), 1, stream));
  std::vector<double> bad_qm = qm_positions;
  bad_qm[1] = std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(fixture.qm_positions.copy_from(bad_qm.data(), bad_qm.size(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_external_point_charge_device_error_cuda(
      device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_external_point_charge_potential_cuda(
      batch, device_potential.get(), device_error.get(), stream));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error ==
        static_cast<std::uint32_t>(Gfn2ExternalPointChargeDeviceError::kNonfiniteQmPosition));

  CUDA_CHECK(fixture.qm_positions.copy_from(qm_positions.data(), qm_positions.size(), stream));
  const std::vector<double> bad_hardness{0.0};
  CUDA_CHECK(fixture.point_hardnesses.copy_from(bad_hardness.data(), 1, stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_external_point_charge_device_error_cuda(
      device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_external_point_charge_potential_cuda(
      batch, device_potential.get(), device_error.get(), stream));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error ==
        static_cast<std::uint32_t>(Gfn2ExternalPointChargeDeviceError::kInvalidPointChargeInput));

  CUDA_CHECK(fixture.point_hardnesses.copy_from(point_hardnesses.data(), 1, stream));
  std::fill(qm_force.begin(), qm_force.end(), 5.0);
  CUDA_CHECK(device_qm_force.copy_from(qm_force.data(), qm_force.size(), stream));
  std::vector<double> bad_shell_charge = shell_charge;
  bad_shell_charge[0] = std::numeric_limits<double>::infinity();
  CUDA_CHECK(
      device_shell_charge.copy_from(bad_shell_charge.data(), bad_shell_charge.size(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_external_point_charge_device_error_cuda(
      device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_energy_cuda(
      batch, device_shell_charge.get(), device_potential.get(), device_energy.get(),
      device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_forces_cuda(
      batch, device_shell_charge.get(), device_qm_force.get(), nullptr, device_error.get(),
      stream));
  CUDA_CHECK(device_qm_force.copy_to(qm_force.data(), qm_force.size(), stream));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error ==
        static_cast<std::uint32_t>(Gfn2ExternalPointChargeDeviceError::kNonfiniteShellValue));
  CHECK(std::all_of(qm_force.begin(), qm_force.end(), [](double value) { return value == 5.0; }));

  /* Finite inputs whose subtraction overflows must not silently write NaN forces. */
  const std::vector<double> extreme_qm{std::numeric_limits<double>::max(), 0.0, 0.0};
  const std::vector<double> extreme_point{-std::numeric_limits<double>::max(), 0.0, 0.0};
  std::vector<double> cpu_potential(static_cast<std::size_t>(plan.total_shells), 17.0);
  std::vector<double> cpu_qm_force(3, 19.0);
  CHECK(gpuxtb::detail::gfn2::evaluate_external_point_charge_potential_cpu(
            plan, extreme_qm.data(), extreme_point.data(), point_charges.data(),
            point_hardnesses.data(), cpu_potential.data(),
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
            plan, extreme_qm.data(), extreme_point.data(), point_charges.data(),
            point_hardnesses.data(), shell_charge.data(), cpu_qm_force.data(), nullptr,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(cpu_qm_force.begin(), cpu_qm_force.end(),
                    [](double value) { return value == 19.0; }));
  CUDA_CHECK(fixture.qm_positions.copy_from(extreme_qm.data(), extreme_qm.size(), stream));
  CUDA_CHECK(fixture.point_positions.copy_from(extreme_point.data(), extreme_point.size(), stream));
  CUDA_CHECK(device_potential.copy_from(cpu_potential.data(), cpu_potential.size(), stream));
  std::fill(qm_force.begin(), qm_force.end(), 5.0);
  CUDA_CHECK(device_qm_force.copy_from(qm_force.data(), qm_force.size(), stream));
  CUDA_CHECK(device_shell_charge.copy_from(shell_charge.data(), shell_charge.size(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_external_point_charge_device_error_cuda(
      device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_external_point_charge_potential_cuda(
      batch, device_potential.get(), device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::add_gfn2_external_point_charge_forces_cuda(
      batch, device_shell_charge.get(), device_qm_force.get(), nullptr, device_error.get(),
      stream));
  CUDA_CHECK(device_qm_force.copy_to(qm_force.data(), qm_force.size(), stream));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error ==
        static_cast<std::uint32_t>(Gfn2ExternalPointChargeDeviceError::kNonfinitePairArithmetic));
  CHECK(std::all_of(qm_force.begin(), qm_force.end(), [](double value) { return value == 5.0; }));

  /* Extreme finite charge/hardness combinations may also overflow the softened potential. */
  const std::vector<double> coincident_qm{0.0, 0.0, 0.0};
  const std::vector<double> coincident_point{0.0, 0.0, 0.0};
  const std::vector<double> extreme_charge{std::numeric_limits<double>::max()};
  const std::vector<double> extreme_hardness{std::numeric_limits<double>::max()};
  CHECK(gpuxtb::detail::gfn2::evaluate_external_point_charge_potential_cpu(
            plan, coincident_qm.data(), coincident_point.data(), extreme_charge.data(),
            extreme_hardness.data(), cpu_potential.data(),
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CUDA_CHECK(fixture.qm_positions.copy_from(coincident_qm.data(), coincident_qm.size(), stream));
  CUDA_CHECK(
      fixture.point_positions.copy_from(coincident_point.data(), coincident_point.size(), stream));
  CUDA_CHECK(fixture.point_charges.copy_from(extreme_charge.data(), 1, stream));
  CUDA_CHECK(fixture.point_hardnesses.copy_from(extreme_hardness.data(), 1, stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_external_point_charge_device_error_cuda(
      device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_external_point_charge_potential_cuda(
      batch, device_potential.get(), device_error.get(), stream));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error ==
        static_cast<std::uint32_t>(Gfn2ExternalPointChargeDeviceError::kNonfinitePairArithmetic));

  invalid = batch;
  invalid.total_atoms = std::numeric_limits<std::int64_t>::max();
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_external_point_charge_potential_cuda(
            invalid, device_potential.get(), device_error.get(), stream) == cudaErrorInvalidValue);
  invalid = batch;
  invalid.batch_size = static_cast<std::int64_t>(std::numeric_limits<int>::max()) + 1;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_external_point_charge_potential_cuda(
            invalid, device_potential.get(), device_error.get(), stream) ==
        cudaErrorInvalidConfiguration);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    if (std::getenv("GPUXTB_TEST_REQUIRE_DEVICE") != nullptr) {
      std::cerr << "CUDA external point-charge test requires a visible device: "
                << (count_status == cudaSuccess ? "none found" : cudaGetErrorString(count_status))
                << '\n';
      return 1;
    }
    std::cout << "CUDA external point-charge test skipped: "
              << (count_status == cudaSuccess ? "no visible device"
                                              : cudaGetErrorString(count_status))
              << '\n';
    return 0;
  }
  if (const int status = test_ragged_water_golden_stream_and_graph(); status != 0) {
    return status;
  }
  if (const int status = test_gated_transactional_force_reduction(); status != 0) {
    return status;
  }
  if (const int status = test_gated_ragged_force_batch_sizes_and_alias_inventory(); status != 0) {
    return status;
  }
  if (const int status = test_zero_total_points_with_null_device_inputs(); status != 0) {
    return status;
  }
  return test_host_and_device_errors_are_sticky();
}
