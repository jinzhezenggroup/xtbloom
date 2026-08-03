#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_energy_force_execution.cuh"
#include "backends/cuda/gfn2_parameters.cuh"
#include "model/gfn2/aes2.hpp"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/coordination.hpp"
#include "model/gfn2/es2.hpp"
#include "model/gfn2/es3.hpp"
#include "model/gfn2/external_point_charges.hpp"
#include "model/gfn2/h0.hpp"
#include "model/gfn2/integrals.hpp"
#include "model/gfn2/repulsion.hpp"
#include "runtime/backend.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using namespace gpuxtb::detail::cuda;
using gpuxtb::detail::gfn2::AES2GeometryCache;
using gpuxtb::detail::gfn2::AES2Plan;
using gpuxtb::detail::gfn2::AES2Workspace;
using gpuxtb::detail::gfn2::BasisPlan;
using gpuxtb::detail::gfn2::CoordinationPlan;
using gpuxtb::detail::gfn2::ES2GeometryCache;
using gpuxtb::detail::gfn2::ES2Plan;
using gpuxtb::detail::gfn2::ES2Workspace;
using gpuxtb::detail::gfn2::ES3Plan;
using gpuxtb::detail::gfn2::ExternalPointChargePlan;
using gpuxtb::detail::gfn2::H0Plan;
using gpuxtb::detail::gfn2::IntegralPlan;
using gpuxtb::detail::gfn2::RepulsionPlan;

constexpr std::uint64_t kPlanToken = 0x671af28de9405cb3ULL;
constexpr std::uint64_t kGeometryGeneration = 17u;
constexpr double kSentinel = -731.25;

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
    return source == nullptr || count > count_
               ? cudaErrorInvalidValue
               : cudaMemcpyAsync(data_, source, count * sizeof(T), cudaMemcpyHostToDevice, stream);
  }
  cudaError_t copy_to(T* target, std::size_t count, cudaStream_t stream) const {
    return target == nullptr || count > count_
               ? cudaErrorInvalidValue
               : cudaMemcpyAsync(target, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
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

bool near(double actual, double expected, double absolute = 6.0e-9, double relative = 6.0e-9) {
  return std::abs(actual - expected) <=
         absolute + relative * std::max(std::abs(actual), std::abs(expected));
}

struct HostCase {
  std::size_t batch_size = 0u;
  BasisPlan basis;
  IntegralPlan integrals;
  H0Plan h0;
  CoordinationPlan coordination_plan;
  ES2Plan es2_plan;
  ES3Plan es3_plan;
  AES2Plan aes2_plan;
  ExternalPointChargePlan external_plan;
  RepulsionPlan repulsion_plan;
  std::int64_t maximum_system_shells = 0;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<std::int64_t> pair_offsets;
  std::vector<std::int64_t> orbital_to_shell;
  std::vector<std::int64_t> orbital_to_atom;

  std::vector<double> positions;
  std::vector<double> point_positions;
  std::vector<double> point_charges;
  std::vector<double> point_hardnesses;
  std::vector<double> external_shell_potential;
  std::vector<double> coordination;
  std::vector<double> overlap;
  std::vector<double> density;
  std::vector<double> weighted_density;
  std::vector<double> shell_scalar;
  std::vector<double> dipole_potential;
  std::vector<double> quadrupole_potential;
  std::vector<double> shell_charges;
  std::vector<double> atomic_charges;
  std::vector<double> atomic_dipoles;
  std::vector<double> atomic_quadrupoles;
  std::vector<double> scc_free_energy;
  std::vector<double> repulsion_energy;

  std::vector<double> es2_matrix;
  std::vector<double> es2_matrix_scratch;
  std::vector<double> es2_shell_scratch;
  std::vector<double> es2_batch_scratch;
  std::vector<double> es2_gradient_scratch;
  ES2Workspace es2_workspace{};
  ES2GeometryCache es2_cache{};

  std::vector<double> aes2_pairs;
  std::vector<double> aes2_pair_scratch;
  std::vector<double> aes2_potential_scratch;
  std::vector<double> aes2_batch_scratch;
  std::vector<double> aes2_gradient_scratch;
  std::vector<double> aes2_coordination_scratch;
  AES2Workspace aes2_workspace{};
  AES2GeometryCache aes2_cache{};

  std::vector<double> expected_energy;
  std::vector<double> expected_electronic_gradient;
  std::vector<double> expected_classical_force;
  std::vector<double> expected_external_qm_force;
  std::vector<double> expected_external_point_force;
  std::vector<double> expected_qm_force;
  std::vector<double> expected_point_force;
};

void add_hamiltonian_adjoints(const HostCase& data, std::vector<double>& overlap_adjoint,
                              std::vector<double>& dipole_adjoint,
                              std::vector<double>& quadrupole_adjoint) {
  const std::int64_t matrices = data.integrals.total_matrix_elements;
  for (std::int64_t system = 0; system < data.basis.batch_size; ++system) {
    const std::size_t s = static_cast<std::size_t>(system);
    const std::int64_t orbital_begin = data.basis.batch_orbital_offsets[s];
    const std::int64_t orbitals = data.basis.batch_orbital_offsets[s + 1u] - orbital_begin;
    const std::int64_t matrix_begin = data.integrals.matrix_offsets[s];
    for (std::int64_t local_row = 0; local_row < orbitals; ++local_row) {
      for (std::int64_t local_column = local_row; local_column < orbitals; ++local_column) {
        const std::int64_t row = orbital_begin + local_row;
        const std::int64_t column = orbital_begin + local_column;
        const std::int64_t row_shell = data.orbital_to_shell[static_cast<std::size_t>(row)];
        const std::int64_t column_shell = data.orbital_to_shell[static_cast<std::size_t>(column)];
        const std::int64_t row_atom = data.orbital_to_atom[static_cast<std::size_t>(row)];
        const std::int64_t column_atom = data.orbital_to_atom[static_cast<std::size_t>(column)];
        const std::int64_t forward = matrix_begin + local_row * orbitals + local_column;
        const std::int64_t reverse = matrix_begin + local_column * orbitals + local_row;
        const double pair_density =
            data.density[static_cast<std::size_t>(forward)] +
            (forward == reverse ? 0.0 : data.density[static_cast<std::size_t>(reverse)]);
        overlap_adjoint[static_cast<std::size_t>(forward)] +=
            -0.5 * pair_density *
            (data.shell_scalar[static_cast<std::size_t>(row_shell)] +
             data.shell_scalar[static_cast<std::size_t>(column_shell)]);
        for (std::int64_t component = 0; component < 3; ++component) {
          dipole_adjoint[static_cast<std::size_t>(component * matrices + forward)] +=
              -0.5 * pair_density *
              data.dipole_potential[static_cast<std::size_t>(3 * column_atom + component)];
          dipole_adjoint[static_cast<std::size_t>(component * matrices + reverse)] +=
              -0.5 * pair_density *
              data.dipole_potential[static_cast<std::size_t>(3 * row_atom + component)];
        }
        for (std::int64_t component = 0; component < 6; ++component) {
          quadrupole_adjoint[static_cast<std::size_t>(component * matrices + forward)] +=
              -0.5 * pair_density *
              data.quadrupole_potential[static_cast<std::size_t>(6 * column_atom + component)];
          quadrupole_adjoint[static_cast<std::size_t>(component * matrices + reverse)] +=
              -0.5 * pair_density *
              data.quadrupole_potential[static_cast<std::size_t>(6 * row_atom + component)];
        }
      }
    }
  }
}

double hamiltonian_shift_energy(const HostCase& data, const std::vector<double>& overlap,
                                const std::vector<double>& dipole,
                                const std::vector<double>& quadrupole, std::size_t system) {
  const std::int64_t orbital_begin = data.basis.batch_orbital_offsets[system];
  const std::int64_t orbitals = data.basis.batch_orbital_offsets[system + 1u] - orbital_begin;
  const std::int64_t matrix_begin = data.integrals.matrix_offsets[system];
  const std::int64_t matrices = data.integrals.total_matrix_elements;
  double energy = 0.0;
  for (std::int64_t row = 0; row < orbitals; ++row) {
    for (std::int64_t column = row; column < orbitals; ++column) {
      const std::int64_t row_global = orbital_begin + row;
      const std::int64_t column_global = orbital_begin + column;
      const std::int64_t row_shell = data.orbital_to_shell[static_cast<std::size_t>(row_global)];
      const std::int64_t column_shell =
          data.orbital_to_shell[static_cast<std::size_t>(column_global)];
      const std::int64_t row_atom = data.orbital_to_atom[static_cast<std::size_t>(row_global)];
      const std::int64_t column_atom =
          data.orbital_to_atom[static_cast<std::size_t>(column_global)];
      const std::int64_t forward = matrix_begin + row * orbitals + column;
      const std::int64_t reverse = matrix_begin + column * orbitals + row;
      double shift = -0.5 * overlap[static_cast<std::size_t>(forward)] *
                     (data.shell_scalar[static_cast<std::size_t>(row_shell)] +
                      data.shell_scalar[static_cast<std::size_t>(column_shell)]);
      for (std::int64_t component = 0; component < 3; ++component) {
        shift -= 0.5 *
                 (dipole[static_cast<std::size_t>(component * matrices + forward)] *
                      data.dipole_potential[static_cast<std::size_t>(3 * column_atom + component)] +
                  dipole[static_cast<std::size_t>(component * matrices + reverse)] *
                      data.dipole_potential[static_cast<std::size_t>(3 * row_atom + component)]);
      }
      for (std::int64_t component = 0; component < 6; ++component) {
        shift -=
            0.5 *
            (quadrupole[static_cast<std::size_t>(component * matrices + forward)] *
                 data.quadrupole_potential[static_cast<std::size_t>(6 * column_atom + component)] +
             quadrupole[static_cast<std::size_t>(component * matrices + reverse)] *
                 data.quadrupole_potential[static_cast<std::size_t>(6 * row_atom + component)]);
      }
      const double pair_density =
          data.density[static_cast<std::size_t>(forward)] +
          (forward == reverse ? 0.0 : data.density[static_cast<std::size_t>(reverse)]);
      energy += pair_density * shift;
    }
  }
  return energy;
}

bool refresh_physics(HostCase& data, std::string& error) {
  const std::size_t atoms = static_cast<std::size_t>(data.basis.total_atoms);
  const std::size_t shells = static_cast<std::size_t>(data.basis.total_shells);
  const std::size_t matrices = static_cast<std::size_t>(data.integrals.total_matrix_elements);
  std::vector<double> cpu_workspace((data.integrals.workspace_size_bytes + sizeof(double) - 1u) /
                                    sizeof(double));
  std::vector<double> dipole_integrals(3u * matrices);
  std::vector<double> quadrupole_integrals(6u * matrices);
  std::vector<double> h0(matrices);
  if (gpuxtb::detail::gfn2::evaluate_coordination_cpu(data.coordination_plan, data.positions.data(),
                                                      data.coordination.data(),
                                                      error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::evaluate_overlap_cpu(data.basis, data.integrals, data.positions.data(),
                                                 data.overlap.data(), cpu_workspace.data(),
                                                 cpu_workspace.size() * sizeof(double),
                                                 error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::evaluate_multipole_cpu(
          data.basis, data.integrals, data.positions.data(), dipole_integrals.data(),
          quadrupole_integrals.data(), cpu_workspace.data(), cpu_workspace.size() * sizeof(double),
          error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::evaluate_h0_cpu(
          data.basis, data.integrals, data.h0, data.positions.data(), data.coordination.data(),
          data.overlap.data(), h0.data(), error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
          data.es2_plan, data.positions.data(), kGeometryGeneration, data.es2_matrix.data(),
          data.es2_matrix.size(), data.es2_workspace, data.es2_cache,
          error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::update_aes2_geometry_cache_cpu(
          data.aes2_plan, data.positions.data(), data.coordination.data(), kGeometryGeneration,
          data.aes2_pairs.data(), data.aes2_pairs.size(), data.aes2_workspace, data.aes2_cache,
          error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }

  std::vector<double> es2_potential(shells);
  std::vector<double> es3_potential(shells);
  std::vector<double> aes2_charge_potential(atoms);
  data.external_shell_potential.resize(shells);
  if (gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
          data.es2_plan, data.es2_cache, data.shell_charges.data(), es2_potential.data(),
          data.es2_workspace, error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::evaluate_es3_potential_cpu(
          gpuxtb::detail::gfn2::make_es3_view(data.es3_plan), data.shell_charges.data(),
          es3_potential.data(), error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::evaluate_aes2_potential_cpu(
          data.aes2_plan, data.aes2_cache, data.atomic_charges.data(), data.atomic_dipoles.data(),
          data.atomic_quadrupoles.data(), aes2_charge_potential.data(),
          data.dipole_potential.data(), data.quadrupole_potential.data(), data.aes2_workspace,
          error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::evaluate_external_point_charge_potential_cpu(
          data.external_plan, data.positions.data(), data.point_positions.data(),
          data.point_charges.data(), data.point_hardnesses.data(),
          data.external_shell_potential.data(), error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  for (std::size_t shell = 0; shell < shells; ++shell) {
    const std::size_t atom =
        static_cast<std::size_t>(data.basis.shell_to_atom[static_cast<std::size_t>(shell)]);
    data.shell_scalar[shell] = es2_potential[shell] + es3_potential[shell] +
                               data.external_shell_potential[shell] + aes2_charge_potential[atom];
  }

  std::vector<double> overlap_adjoint(matrices, 0.0);
  std::vector<double> coordination_adjoint(atoms, 0.0);
  std::vector<double> dipole_adjoint(3u * matrices, 0.0);
  std::vector<double> quadrupole_adjoint(6u * matrices, 0.0);
  data.expected_electronic_gradient.assign(3u * atoms, 0.0);
  if (gpuxtb::detail::gfn2::add_h0_vjp_cpu(
          data.basis, data.integrals, data.h0, data.positions.data(), data.coordination.data(),
          data.overlap.data(), data.density.data(), overlap_adjoint.data(),
          coordination_adjoint.data(), data.expected_electronic_gradient.data(),
          error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  for (std::size_t matrix = 0; matrix < matrices; ++matrix) {
    overlap_adjoint[matrix] -= data.weighted_density[matrix];
  }
  add_hamiltonian_adjoints(data, overlap_adjoint, dipole_adjoint, quadrupole_adjoint);
  if (gpuxtb::detail::gfn2::add_overlap_gradient_cpu(
          data.basis, data.integrals, data.positions.data(), overlap_adjoint.data(),
          data.expected_electronic_gradient.data(), cpu_workspace.data(),
          cpu_workspace.size() * sizeof(double), error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::add_multipole_gradient_cpu(
          data.basis, data.integrals, data.positions.data(), dipole_adjoint.data(),
          quadrupole_adjoint.data(), data.expected_electronic_gradient.data(), cpu_workspace.data(),
          cpu_workspace.size() * sizeof(double), error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::add_coordination_gradient_cpu(
          data.coordination_plan, data.positions.data(), coordination_adjoint.data(),
          data.expected_electronic_gradient.data(), error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }

  data.repulsion_energy.assign(data.batch_size, 0.0);
  data.expected_classical_force.assign(3u * atoms, 0.0);
  if (gpuxtb::detail::gfn2::add_repulsion_cpu(
          data.repulsion_plan, data.positions.data(), data.repulsion_energy.data(),
          data.expected_classical_force.data(), error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  std::vector<double> classical_gradient(3u * atoms, 0.0);
  std::vector<double> aes2_coordination_adjoint(atoms, 0.0);
  if (gpuxtb::detail::gfn2::add_es2_gradient_cpu(
          data.es2_plan, data.es2_cache, data.positions.data(), kGeometryGeneration,
          data.shell_charges.data(), classical_gradient.data(), data.es2_workspace,
          error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::add_aes2_vjp_cpu(
          data.aes2_plan, data.aes2_cache, data.positions.data(), data.coordination.data(),
          kGeometryGeneration, data.atomic_charges.data(), data.atomic_dipoles.data(),
          data.atomic_quadrupoles.data(), classical_gradient.data(),
          aes2_coordination_adjoint.data(), data.aes2_workspace, error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::add_coordination_gradient_cpu(
          data.coordination_plan, data.positions.data(), aes2_coordination_adjoint.data(),
          classical_gradient.data(), error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  for (std::size_t coordinate = 0; coordinate < classical_gradient.size(); ++coordinate) {
    data.expected_classical_force[coordinate] -= classical_gradient[coordinate];
  }

  std::vector<double> external_energy(data.batch_size, 0.0);
  data.expected_external_qm_force.assign(3u * atoms, 0.0);
  data.expected_external_point_force.assign(data.point_positions.size(), 0.0);
  if (gpuxtb::detail::gfn2::add_external_point_charge_energy_cpu(
          data.external_plan, data.shell_charges.data(), data.external_shell_potential.data(),
          external_energy.data(), error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::add_external_point_charge_forces_cpu(
          data.external_plan, data.positions.data(), data.point_positions.data(),
          data.point_charges.data(), data.point_hardnesses.data(), data.shell_charges.data(),
          data.expected_external_qm_force.data(), data.expected_external_point_force.data(),
          error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }

  std::vector<double> es2_energy(data.batch_size, 0.0);
  std::vector<double> es3_energy(data.batch_size, 0.0);
  std::vector<double> aes2_energy(data.batch_size, 0.0);
  if (gpuxtb::detail::gfn2::add_es2_energy_cpu(
          data.es2_plan, data.es2_cache, data.shell_charges.data(), es2_energy.data(),
          data.es2_workspace, error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::add_es3_energy_cpu(gpuxtb::detail::gfn2::make_es3_view(data.es3_plan),
                                               data.shell_charges.data(), es3_energy.data(),
                                               error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::add_aes2_energy_cpu(
          data.aes2_plan, data.aes2_cache, data.atomic_charges.data(), data.atomic_dipoles.data(),
          data.atomic_quadrupoles.data(), aes2_energy.data(), data.aes2_workspace,
          error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }

  data.scc_free_energy.assign(data.batch_size, 0.0);
  for (std::size_t system = 0; system < data.batch_size; ++system) {
    const std::size_t matrix_begin =
        static_cast<std::size_t>(data.integrals.matrix_offsets[system]);
    const std::size_t matrix_end =
        static_cast<std::size_t>(data.integrals.matrix_offsets[system + 1u]);
    double core = 0.0;
    for (std::size_t matrix = matrix_begin; matrix < matrix_end; ++matrix) {
      core +=
          data.density[matrix] * h0[matrix] - data.weighted_density[matrix] * data.overlap[matrix];
    }
    data.scc_free_energy[system] = core +
                                   hamiltonian_shift_energy(data, data.overlap, dipole_integrals,
                                                            quadrupole_integrals, system) +
                                   es2_energy[system] + es3_energy[system] + aes2_energy[system] +
                                   external_energy[system];
  }
  data.expected_energy.resize(data.batch_size);
  for (std::size_t system = 0; system < data.batch_size; ++system) {
    data.expected_energy[system] = data.scc_free_energy[system] + data.repulsion_energy[system];
  }
  data.expected_qm_force.resize(3u * atoms);
  for (std::size_t coordinate = 0; coordinate < data.expected_qm_force.size(); ++coordinate) {
    data.expected_qm_force[coordinate] = -data.expected_electronic_gradient[coordinate] +
                                         data.expected_classical_force[coordinate] +
                                         data.expected_external_qm_force[coordinate];
  }
  data.expected_point_force = data.expected_external_point_force;
  return true;
}

bool make_case(std::size_t batch_size, HostCase& data, std::string& error) {
  data.batch_size = batch_size;
  std::vector<std::int64_t> atom_offsets(batch_size + 1u);
  std::vector<std::int64_t> point_offsets(batch_size + 1u);
  data.atomic_numbers.assign(2u * batch_size, 1);
  data.positions.resize(6u * batch_size);
  data.point_positions.resize(3u * batch_size);
  data.point_charges.resize(batch_size);
  data.point_hardnesses.resize(batch_size);
  for (std::size_t system = 0; system < batch_size; ++system) {
    atom_offsets[system] = static_cast<std::int64_t>(2u * system);
    point_offsets[system] = static_cast<std::int64_t>(system);
    const double origin = 4.0 * static_cast<double>(system);
    const double distance = 1.20 + 0.01 * static_cast<double>(system % 7u);
    data.positions[6u * system] = origin - 0.11;
    data.positions[6u * system + 1u] = 0.07;
    data.positions[6u * system + 2u] = -0.5 * distance;
    data.positions[6u * system + 3u] = origin + 0.13;
    data.positions[6u * system + 4u] = -0.05;
    data.positions[6u * system + 5u] = 0.5 * distance;
    data.point_positions[3u * system] = origin + 0.41;
    data.point_positions[3u * system + 1u] = 0.53;
    data.point_positions[3u * system + 2u] = -0.37;
    data.point_charges[system] = system % 2u == 0u ? 0.37 : -0.29;
    data.point_hardnesses[system] = 0.61 + 0.01 * static_cast<double>(system % 5u);
  }
  atom_offsets[batch_size] = static_cast<std::int64_t>(2u * batch_size);
  point_offsets[batch_size] = static_cast<std::int64_t>(batch_size);
  if (gpuxtb::detail::gfn2::make_basis_plan(static_cast<std::int64_t>(batch_size),
                                            static_cast<std::int64_t>(2u * batch_size),
                                            atom_offsets.data(), data.atomic_numbers.data(),
                                            data.basis, error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::make_integral_plan(data.basis, data.integrals, error) !=
          GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::make_h0_plan(data.basis, data.integrals, data.atomic_numbers.data(),
                                         data.h0, error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::make_coordination_plan(
          static_cast<std::int64_t>(batch_size), static_cast<std::int64_t>(2u * batch_size),
          atom_offsets.data(), data.atomic_numbers.data(), data.coordination_plan,
          error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::make_es2_plan(data.basis, data.atomic_numbers.data(), data.es2_plan,
                                          error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::make_es3_plan(data.basis, data.atomic_numbers.data(), data.es3_plan,
                                          error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::make_aes2_plan(data.basis, data.atomic_numbers.data(), data.aes2_plan,
                                           error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::make_repulsion_plan(
          static_cast<std::int64_t>(batch_size), static_cast<std::int64_t>(2u * batch_size),
          atom_offsets.data(), data.atomic_numbers.data(), data.repulsion_plan,
          error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::make_external_point_charge_plan(
          data.basis, data.atomic_numbers.data(), static_cast<std::int64_t>(batch_size),
          point_offsets.data(), data.external_plan, error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  data.pair_offsets.resize(batch_size + 1u);
  for (std::size_t system = 0; system <= batch_size; ++system) {
    data.pair_offsets[system] = static_cast<std::int64_t>(system);
  }
  for (std::size_t system = 0; system < batch_size; ++system) {
    const std::int64_t shells =
        data.basis.batch_shell_offsets[system + 1u] - data.basis.batch_shell_offsets[system];
    data.maximum_system_shells = std::max(data.maximum_system_shells, shells);
  }
  data.orbital_to_shell.resize(static_cast<std::size_t>(data.basis.total_orbitals));
  data.orbital_to_atom.resize(static_cast<std::size_t>(data.basis.total_orbitals));
  for (std::int64_t shell = 0; shell < data.basis.total_shells; ++shell) {
    for (std::int64_t orbital = data.basis.shell_orbital_offsets[static_cast<std::size_t>(shell)];
         orbital < data.basis.shell_orbital_offsets[static_cast<std::size_t>(shell + 1)];
         ++orbital) {
      data.orbital_to_shell[static_cast<std::size_t>(orbital)] = shell;
      data.orbital_to_atom[static_cast<std::size_t>(orbital)] =
          data.basis.shell_to_atom[static_cast<std::size_t>(shell)];
    }
  }
  const std::size_t atoms = static_cast<std::size_t>(data.basis.total_atoms);
  const std::size_t shells = static_cast<std::size_t>(data.basis.total_shells);
  const std::size_t matrices = static_cast<std::size_t>(data.integrals.total_matrix_elements);
  data.coordination.resize(atoms);
  data.overlap.resize(matrices);
  data.density.resize(matrices);
  data.weighted_density.resize(matrices);
  data.shell_scalar.resize(shells);
  data.dipole_potential.resize(3u * atoms);
  data.quadrupole_potential.resize(6u * atoms);
  data.shell_charges.resize(shells);
  data.atomic_charges.assign(atoms, 0.0);
  data.atomic_dipoles.resize(3u * atoms);
  data.atomic_quadrupoles.resize(6u * atoms);
  for (std::int64_t system = 0; system < data.basis.batch_size; ++system) {
    const std::size_t s = static_cast<std::size_t>(system);
    const std::int64_t begin = data.integrals.matrix_offsets[s];
    const std::int64_t orbitals =
        data.basis.batch_orbital_offsets[s + 1u] - data.basis.batch_orbital_offsets[s];
    for (std::int64_t row = 0; row < orbitals; ++row) {
      for (std::int64_t column = 0; column <= row; ++column) {
        const double density = 0.18 + 0.01 * static_cast<double>((row + 2 * column + system) % 7);
        const double weighted =
            -0.24 + 0.008 * static_cast<double>((2 * row + column + system) % 9);
        data.density[static_cast<std::size_t>(begin + row * orbitals + column)] = density;
        data.density[static_cast<std::size_t>(begin + column * orbitals + row)] = density;
        data.weighted_density[static_cast<std::size_t>(begin + row * orbitals + column)] = weighted;
        data.weighted_density[static_cast<std::size_t>(begin + column * orbitals + row)] = weighted;
      }
    }
  }
  for (std::size_t shell = 0; shell < shells; ++shell) {
    data.shell_charges[shell] = 0.018 * static_cast<double>(static_cast<int>(shell % 7u) - 3);
    const std::size_t atom =
        static_cast<std::size_t>(data.basis.shell_to_atom[static_cast<std::size_t>(shell)]);
    data.atomic_charges[atom] += data.shell_charges[shell];
  }
  for (std::size_t element = 0; element < data.atomic_dipoles.size(); ++element) {
    data.atomic_dipoles[element] = 0.004 * static_cast<double>(static_cast<int>(element % 7u) - 3);
  }
  for (std::size_t element = 0; element < data.atomic_quadrupoles.size(); ++element) {
    data.atomic_quadrupoles[element] =
        0.0015 * static_cast<double>(static_cast<int>(element % 11u) - 5);
  }

  data.es2_matrix.resize(static_cast<std::size_t>(data.es2_plan.total_matrix_elements()));
  data.es2_matrix_scratch.resize(data.es2_matrix.size());
  data.es2_shell_scratch.resize(shells);
  data.es2_batch_scratch.resize(batch_size);
  data.es2_gradient_scratch.resize(3u * atoms);
  data.es2_workspace = {data.es2_matrix_scratch.data(),   data.es2_plan.total_matrix_elements(),
                        data.es2_shell_scratch.data(),    data.es2_plan.total_shells(),
                        data.es2_batch_scratch.data(),    static_cast<std::int64_t>(batch_size),
                        data.es2_gradient_scratch.data(), static_cast<std::int64_t>(3u * atoms)};

  data.aes2_pairs.resize(static_cast<std::size_t>(data.aes2_plan.pair_data_elements()));
  data.aes2_pair_scratch.resize(data.aes2_pairs.size());
  data.aes2_potential_scratch.resize(
      static_cast<std::size_t>(data.aes2_plan.potential_scratch_elements()));
  data.aes2_batch_scratch.resize(batch_size);
  data.aes2_gradient_scratch.resize(3u * atoms);
  data.aes2_coordination_scratch.resize(atoms);
  data.aes2_workspace = {
      data.aes2_pair_scratch.data(),         data.aes2_plan.pair_data_elements(),
      data.aes2_potential_scratch.data(),    data.aes2_plan.potential_scratch_elements(),
      data.aes2_batch_scratch.data(),        static_cast<std::int64_t>(batch_size),
      data.aes2_gradient_scratch.data(),     static_cast<std::int64_t>(3u * atoms),
      data.aes2_coordination_scratch.data(), static_cast<std::int64_t>(atoms)};
  return refresh_physics(data, error);
}

struct DeviceFixture {
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> pair_offsets;
  DeviceBuffer<std::int64_t> batch_shell_offsets;
  DeviceBuffer<std::int64_t> qsh_offsets;
  DeviceBuffer<std::int64_t> batch_orbital_offsets;
  DeviceBuffer<std::int64_t> integral_matrix_offsets;
  DeviceBuffer<std::int64_t> es2_matrix_offsets;
  DeviceBuffer<std::int64_t> shell_pair_offsets;
  DeviceBuffer<std::int64_t> atom_shell_offsets;
  DeviceBuffer<std::int64_t> shell_orbital_offsets;
  DeviceBuffer<std::int64_t> shell_primitive_offsets;
  DeviceBuffer<std::int64_t> shell_to_atom;
  DeviceBuffer<std::int64_t> orbital_to_shell;
  DeviceBuffer<std::int64_t> orbital_to_atom;
  DeviceBuffer<std::int64_t> point_offsets;
  DeviceBuffer<std::int64_t> dipole_offsets;
  DeviceBuffer<std::int64_t> quadrupole_offsets;
  DeviceBuffer<std::uint8_t> angular_momenta;
  DeviceBuffer<std::int32_t> atomic_numbers;

  DeviceBuffer<double> primitive_exponents;
  DeviceBuffer<double> primitive_coefficients;
  DeviceBuffer<double> atomic_radii;
  DeviceBuffer<double> shell_levels;
  DeviceBuffer<double> shell_coordination_scale;
  DeviceBuffer<double> shell_polynomial;
  DeviceBuffer<double> shell_pair_scale;
  DeviceBuffer<double> covalent_radii;
  DeviceBuffer<double> es2_hardness;
  DeviceBuffer<double> es3_gamma;
  DeviceBuffer<double> aes2_dipole_kernel;
  DeviceBuffer<double> aes2_quadrupole_kernel;
  DeviceBuffer<double> aes2_radius;
  DeviceBuffer<double> aes2_valence_cn;
  DeviceBuffer<double> repulsion_sqrt_alpha;
  DeviceBuffer<double> repulsion_effective_charge;
  DeviceBuffer<std::uint8_t> repulsion_light_element;
  DeviceBuffer<double> external_shell_hardness;

  DeviceBuffer<double> positions;
  DeviceBuffer<double> point_positions;
  DeviceBuffer<double> point_charges;
  DeviceBuffer<double> point_hardnesses;
  DeviceBuffer<double> external_shell_potential;
  DeviceBuffer<double> coordination;
  DeviceBuffer<double> overlap;
  DeviceBuffer<double> density;
  DeviceBuffer<double> weighted_density;
  DeviceBuffer<double> shell_scalar;
  DeviceBuffer<double> dipole_potential;
  DeviceBuffer<double> quadrupole_potential;
  DeviceBuffer<double> shell_charges;
  DeviceBuffer<double> atomic_charges;
  DeviceBuffer<double> atomic_dipoles;
  DeviceBuffer<double> atomic_quadrupoles;
  DeviceBuffer<double> scc_free_energy;
  DeviceBuffer<double> repulsion_energy;

  DeviceBuffer<double> geometry_pairs;
  DeviceBuffer<double> geometry_coordination;
  DeviceBuffer<std::uint64_t> geometry_generations;
  DeviceBuffer<double> geometry_pair_scratch;
  DeviceBuffer<double> geometry_coordination_scratch;
  DeviceBuffer<double> geometry_update_gradient_scratch;
  DeviceBuffer<std::uint32_t> geometry_update_sequence;
  DeviceBuffer<double> es2_matrix;
  DeviceBuffer<double> es2_matrix_scratch;
  DeviceBuffer<double> aes2_pairs;
  DeviceBuffer<double> aes2_pair_scratch;
  DeviceBuffer<std::uint32_t> aes2_peer_error_scratch;

  /* Fresh raw post-SCC potentials and unpublished component staging. */
  DeviceBuffer<double> post_complete_shell;
  DeviceBuffer<double> post_complete_atomic;
  DeviceBuffer<double> post_es2_shell;
  DeviceBuffer<double> post_es3_shell;
  DeviceBuffer<double> post_aes2_atomic;
  DeviceBuffer<double> post_aes2_dipole;
  DeviceBuffer<double> post_aes2_quadrupole;
  DeviceBuffer<double> post_staged_shell;
  DeviceBuffer<double> post_staged_atomic;
  DeviceBuffer<double> post_staged_dipole;
  DeviceBuffer<double> post_staged_quadrupole;
  DeviceBuffer<double> post_staged_shell_scalar;
  DeviceBuffer<double> post_es2_shell_scratch;
  DeviceBuffer<double> post_aes2_potential_scratch;
  DeviceBuffer<std::uint32_t> post_aes2_peer_scratch;
  DeviceBuffer<double> post_compose_shell_scratch;
  DeviceBuffer<double> post_compose_atomic_scratch;
  DeviceBuffer<double> post_compose_dipole_scratch;
  DeviceBuffer<double> post_compose_quadrupole_scratch;
  DeviceBuffer<double> post_bridge_scratch;

  DeviceBuffer<double> staged_energy;
  DeviceBuffer<double> public_energy;
  DeviceBuffer<double> overlap_adjoint;
  DeviceBuffer<double> coordination_adjoint;
  DeviceBuffer<double> dipole_adjoint;
  DeviceBuffer<double> quadrupole_adjoint;
  DeviceBuffer<double> electronic_gradient;
  DeviceBuffer<double> classical_force;
  DeviceBuffer<double> explicit_qm_force;
  DeviceBuffer<double> explicit_point_force;
  DeviceBuffer<double> staged_qm_force;
  DeviceBuffer<double> staged_point_force;
  DeviceBuffer<double> public_qm_force;
  DeviceBuffer<double> public_point_force;

  DeviceBuffer<double> h0_overlap_scratch;
  DeviceBuffer<double> h0_coordination_scratch;
  DeviceBuffer<double> h0_gradient_scratch;
  DeviceBuffer<double> hamiltonian_overlap_scratch;
  DeviceBuffer<double> hamiltonian_dipole_scratch;
  DeviceBuffer<double> hamiltonian_quadrupole_scratch;
  DeviceBuffer<double> integral_gradient_scratch;
  DeviceBuffer<double> coordination_gradient_scratch;
  DeviceBuffer<double> classical_gradient_scratch;
  DeviceBuffer<double> classical_force_scratch;
  DeviceBuffer<double> classical_coordination_adjoint;
  DeviceBuffer<double> classical_aes2_gradient_scratch;
  DeviceBuffer<double> classical_aes2_coordination_scratch;
  DeviceBuffer<double> classical_geometry_gradient_scratch;
  DeviceBuffer<double> external_qm_scratch;
  DeviceBuffer<double> external_point_scratch;
  DeviceBuffer<double> composition_qm_scratch;
  DeviceBuffer<double> composition_point_scratch;

  DeviceBuffer<std::uint8_t> requested;
  DeviceBuffer<gpuxtb_status_t> statuses;
  DeviceBuffer<std::uint8_t> converged;
  DeviceBuffer<std::uint8_t> energy_success;
  DeviceBuffer<std::uint8_t> post_scc_success;
  DeviceBuffer<std::uint8_t> post_active;
  DeviceBuffer<std::uint8_t> electronic_success;
  DeviceBuffer<std::uint8_t> coordination_success;
  DeviceBuffer<std::uint8_t> classical_success;
  DeviceBuffer<std::uint8_t> external_success;
  DeviceBuffer<std::uint8_t> h0_success;
  DeviceBuffer<std::uint8_t> hamiltonian_success;
  DeviceBuffer<std::uint8_t> classical_selected;

  DeviceBuffer<std::uint32_t> total_sequence;
  DeviceBuffer<std::uint32_t> post_sequence;
  DeviceBuffer<std::uint32_t> post_compose_sequence;
  DeviceBuffer<std::uint32_t> post_bridge_sequence;
  DeviceBuffer<std::uint32_t> h0_sequence;
  DeviceBuffer<std::uint32_t> hamiltonian_sequence;
  DeviceBuffer<std::uint32_t> integral_sequence;
  DeviceBuffer<std::uint32_t> coordination_sequence;
  DeviceBuffer<std::uint32_t> classical_sequence;
  DeviceBuffer<std::uint32_t> classical_primitive_sequence;
  DeviceBuffer<std::uint32_t> external_sequence;
  DeviceBuffer<std::uint32_t> composition_sequence;
  DeviceBuffer<std::uint32_t> plan_failure;
  DeviceBuffer<std::uint32_t> classical_primitive_system_errors;
  DeviceBuffer<std::uint32_t> classical_primitive_device_error;
  DeviceBuffer<std::uint32_t> post_stage_system_errors;
  DeviceBuffer<std::uint32_t> post_stage_device_error;
  DeviceBuffer<std::uint32_t> post_system_errors;
  DeviceBuffer<std::uint32_t> post_device_error;

  DeviceBuffer<std::uint32_t> execution_system_errors;
  DeviceBuffer<std::uint32_t> execution_device_error;
  DeviceBuffer<std::uint32_t> total_system_errors;
  DeviceBuffer<std::uint32_t> total_device_error;
  DeviceBuffer<std::uint32_t> h0_system_errors;
  DeviceBuffer<std::uint32_t> h0_device_error;
  DeviceBuffer<std::uint32_t> hamiltonian_system_errors;
  DeviceBuffer<std::uint32_t> hamiltonian_device_error;
  DeviceBuffer<std::uint32_t> integral_system_errors;
  DeviceBuffer<std::uint32_t> integral_device_error;
  DeviceBuffer<std::uint32_t> coordination_system_errors;
  DeviceBuffer<std::uint32_t> coordination_device_error;
  DeviceBuffer<std::uint32_t> classical_system_errors;
  DeviceBuffer<std::uint32_t> classical_device_error;
  DeviceBuffer<std::uint32_t> external_system_errors;
  DeviceBuffer<std::uint32_t> external_device_error;
  DeviceBuffer<std::uint32_t> composition_system_errors;
  DeviceBuffer<std::uint32_t> composition_plan_error;

  Gfn2EnergyForceExecutionDevicePlan plan{};
  Gfn2EnergyForceExecutionDeviceInput input{};
  Gfn2EnergyForceExecutionDeviceResults results{};
  Gfn2EnergyForceExecutionDeviceIntermediates intermediates{};
  Gfn2EnergyForceExecutionDeviceWorkspace workspace{};
  Gfn2EnergyForceExecutionDeviceDiagnostics diagnostics{};
};

cudaError_t initialize_device(DeviceFixture& d, const HostCase& h, cudaStream_t stream) {
  std::vector<std::int64_t> dipole_offsets(h.basis.atom_offsets.size());
  std::vector<std::int64_t> quadrupole_offsets(h.basis.atom_offsets.size());
  for (std::size_t system = 0; system < h.basis.atom_offsets.size(); ++system) {
    dipole_offsets[system] = 3 * h.basis.atom_offsets[system];
    quadrupole_offsets[system] = 6 * h.basis.atom_offsets[system];
  }
  cudaError_t status = upload(d.atom_offsets, h.basis.atom_offsets, stream);
#define UPLOAD(field, source)                   \
  if (status == cudaSuccess) {                  \
    status = upload(d.field, (source), stream); \
  }
  UPLOAD(pair_offsets, h.pair_offsets)
  UPLOAD(batch_shell_offsets, h.basis.batch_shell_offsets)
  UPLOAD(qsh_offsets, h.basis.batch_shell_offsets)
  UPLOAD(batch_orbital_offsets, h.basis.batch_orbital_offsets)
  UPLOAD(integral_matrix_offsets, h.integrals.matrix_offsets)
  UPLOAD(es2_matrix_offsets, h.es2_plan.matrix_offsets())
  UPLOAD(shell_pair_offsets, h.h0.shell_pair_offsets)
  UPLOAD(atom_shell_offsets, h.basis.atom_shell_offsets)
  UPLOAD(shell_orbital_offsets, h.basis.shell_orbital_offsets)
  UPLOAD(shell_primitive_offsets, h.basis.shell_primitive_offsets)
  UPLOAD(shell_to_atom, h.basis.shell_to_atom)
  UPLOAD(orbital_to_shell, h.orbital_to_shell)
  UPLOAD(orbital_to_atom, h.orbital_to_atom)
  UPLOAD(point_offsets, h.external_plan.point_charge_offsets)
  UPLOAD(dipole_offsets, dipole_offsets)
  UPLOAD(quadrupole_offsets, quadrupole_offsets)
  UPLOAD(angular_momenta, h.basis.angular_momenta)
  UPLOAD(atomic_numbers, h.atomic_numbers)
  UPLOAD(primitive_exponents, h.basis.primitive_exponents)
  UPLOAD(primitive_coefficients, h.basis.primitive_coefficients)
  UPLOAD(atomic_radii, h.h0.atomic_radii)
  UPLOAD(shell_levels, h.h0.shell_levels)
  UPLOAD(shell_coordination_scale, h.h0.shell_coordination_scale)
  UPLOAD(shell_polynomial, h.h0.shell_polynomial)
  UPLOAD(shell_pair_scale, h.h0.shell_pair_scale)
  UPLOAD(covalent_radii, h.coordination_plan.covalent_radius)
  UPLOAD(es2_hardness, h.es2_plan.shell_hardness())
  UPLOAD(es3_gamma, h.es3_plan.shell_gamma3)
  UPLOAD(aes2_dipole_kernel, h.aes2_plan.dipole_kernel())
  UPLOAD(aes2_quadrupole_kernel, h.aes2_plan.quadrupole_kernel())
  UPLOAD(aes2_radius, h.aes2_plan.multipole_radius())
  UPLOAD(aes2_valence_cn, h.aes2_plan.multipole_valence_cn())
  UPLOAD(repulsion_sqrt_alpha, h.repulsion_plan.sqrt_alpha)
  UPLOAD(repulsion_effective_charge, h.repulsion_plan.effective_charge)
  UPLOAD(repulsion_light_element, h.repulsion_plan.light_element)
  UPLOAD(external_shell_hardness, h.external_plan.shell_hardness)
  UPLOAD(external_shell_potential, h.external_shell_potential)
#undef UPLOAD
  if (status != cudaSuccess) {
    return status;
  }

  const std::size_t batch = h.batch_size;
  const std::size_t atoms = static_cast<std::size_t>(h.basis.total_atoms);
  const std::size_t shells = static_cast<std::size_t>(h.basis.total_shells);
  const std::size_t matrices = static_cast<std::size_t>(h.integrals.total_matrix_elements);
  const std::size_t pairs = static_cast<std::size_t>(h.aes2_plan.total_pairs());
  const std::size_t coordinates = atoms * 3u;
  const std::size_t point_coordinates = h.point_positions.size();
#define ALLOCATE(field, count)          \
  if (status == cudaSuccess) {          \
    status = d.field.allocate((count)); \
  }
  ALLOCATE(positions, coordinates)
  ALLOCATE(point_positions, point_coordinates)
  ALLOCATE(point_charges, batch)
  ALLOCATE(point_hardnesses, batch)
  ALLOCATE(coordination, atoms)
  ALLOCATE(overlap, matrices)
  ALLOCATE(density, matrices)
  ALLOCATE(weighted_density, matrices)
  ALLOCATE(shell_scalar, shells)
  ALLOCATE(dipole_potential, coordinates)
  ALLOCATE(quadrupole_potential, atoms * 6u)
  ALLOCATE(shell_charges, shells)
  ALLOCATE(atomic_charges, atoms)
  ALLOCATE(atomic_dipoles, coordinates)
  ALLOCATE(atomic_quadrupoles, atoms * 6u)
  ALLOCATE(scc_free_energy, batch)
  ALLOCATE(repulsion_energy, batch)
  ALLOCATE(geometry_pairs, pairs * static_cast<std::size_t>(kGfn2GeometryPairDataElements))
  ALLOCATE(geometry_coordination, atoms)
  ALLOCATE(geometry_generations, batch)
  ALLOCATE(geometry_pair_scratch, pairs * static_cast<std::size_t>(kGfn2GeometryPairDataElements))
  ALLOCATE(geometry_coordination_scratch, atoms)
  ALLOCATE(geometry_update_gradient_scratch, coordinates)
  ALLOCATE(geometry_update_sequence, 1u)
  ALLOCATE(es2_matrix, h.es2_matrix.size())
  ALLOCATE(es2_matrix_scratch, h.es2_matrix.size())
  ALLOCATE(aes2_pairs, h.aes2_pairs.size())
  ALLOCATE(aes2_pair_scratch, h.aes2_pairs.size())
  ALLOCATE(aes2_peer_error_scratch, 1u)
  ALLOCATE(post_complete_shell, shells)
  ALLOCATE(post_complete_atomic, atoms)
  ALLOCATE(post_es2_shell, shells)
  ALLOCATE(post_es3_shell, shells)
  ALLOCATE(post_aes2_atomic, atoms)
  ALLOCATE(post_aes2_dipole, coordinates)
  ALLOCATE(post_aes2_quadrupole, atoms * 6u)
  ALLOCATE(post_staged_shell, shells)
  ALLOCATE(post_staged_atomic, atoms)
  ALLOCATE(post_staged_dipole, coordinates)
  ALLOCATE(post_staged_quadrupole, atoms * 6u)
  ALLOCATE(post_staged_shell_scalar, shells)
  ALLOCATE(post_es2_shell_scratch, shells)
  ALLOCATE(post_aes2_potential_scratch, h.aes2_plan.potential_scratch_elements())
  ALLOCATE(post_aes2_peer_scratch, 1u)
  ALLOCATE(post_compose_shell_scratch, shells)
  ALLOCATE(post_compose_atomic_scratch, atoms)
  ALLOCATE(post_compose_dipole_scratch, coordinates)
  ALLOCATE(post_compose_quadrupole_scratch, atoms * 6u)
  ALLOCATE(post_bridge_scratch, shells)
  ALLOCATE(staged_energy, batch)
  ALLOCATE(public_energy, batch)
  ALLOCATE(overlap_adjoint, matrices)
  ALLOCATE(coordination_adjoint, atoms)
  ALLOCATE(dipole_adjoint, 3u * matrices)
  ALLOCATE(quadrupole_adjoint, 6u * matrices)
  ALLOCATE(electronic_gradient, coordinates)
  ALLOCATE(classical_force, coordinates)
  ALLOCATE(explicit_qm_force, coordinates)
  ALLOCATE(explicit_point_force, point_coordinates)
  ALLOCATE(staged_qm_force, coordinates)
  ALLOCATE(staged_point_force, point_coordinates)
  ALLOCATE(public_qm_force, coordinates)
  ALLOCATE(public_point_force, point_coordinates)
  ALLOCATE(h0_overlap_scratch, matrices)
  ALLOCATE(h0_coordination_scratch, atoms)
  ALLOCATE(h0_gradient_scratch, coordinates)
  ALLOCATE(hamiltonian_overlap_scratch, matrices)
  ALLOCATE(hamiltonian_dipole_scratch, 3u * matrices)
  ALLOCATE(hamiltonian_quadrupole_scratch, 6u * matrices)
  ALLOCATE(integral_gradient_scratch, coordinates)
  ALLOCATE(coordination_gradient_scratch, coordinates)
  ALLOCATE(classical_gradient_scratch, coordinates)
  ALLOCATE(classical_force_scratch, coordinates)
  ALLOCATE(classical_coordination_adjoint, atoms)
  ALLOCATE(classical_aes2_gradient_scratch, coordinates)
  ALLOCATE(classical_aes2_coordination_scratch, atoms)
  ALLOCATE(classical_geometry_gradient_scratch, coordinates)
  ALLOCATE(external_qm_scratch, coordinates)
  ALLOCATE(external_point_scratch, point_coordinates)
  ALLOCATE(composition_qm_scratch, coordinates)
  ALLOCATE(composition_point_scratch, point_coordinates)
  ALLOCATE(requested, batch)
  ALLOCATE(statuses, batch)
  ALLOCATE(converged, batch)
  ALLOCATE(energy_success, batch)
  ALLOCATE(post_scc_success, batch)
  ALLOCATE(post_active, batch)
  ALLOCATE(electronic_success, batch)
  ALLOCATE(coordination_success, batch)
  ALLOCATE(classical_success, batch)
  ALLOCATE(external_success, batch)
  ALLOCATE(h0_success, batch)
  ALLOCATE(hamiltonian_success, batch)
  ALLOCATE(classical_selected, batch)
  ALLOCATE(total_sequence, 1u)
  ALLOCATE(post_sequence, 1u)
  ALLOCATE(post_compose_sequence, 1u)
  ALLOCATE(post_bridge_sequence, 1u)
  ALLOCATE(h0_sequence, 1u)
  ALLOCATE(hamiltonian_sequence, 1u)
  ALLOCATE(integral_sequence, 1u)
  ALLOCATE(coordination_sequence, 1u)
  ALLOCATE(classical_sequence, 1u)
  ALLOCATE(classical_primitive_sequence, 1u)
  ALLOCATE(external_sequence, 1u)
  ALLOCATE(composition_sequence, 1u)
  ALLOCATE(plan_failure, 1u)
  ALLOCATE(classical_primitive_system_errors, batch)
  ALLOCATE(classical_primitive_device_error, 1u)
  ALLOCATE(post_stage_system_errors, batch)
  ALLOCATE(post_stage_device_error, 1u)
  ALLOCATE(post_system_errors, batch)
  ALLOCATE(post_device_error, 1u)
  ALLOCATE(execution_system_errors, batch)
  ALLOCATE(execution_device_error, 1u)
  ALLOCATE(total_system_errors, batch)
  ALLOCATE(total_device_error, 1u)
  ALLOCATE(h0_system_errors, batch)
  ALLOCATE(h0_device_error, 1u)
  ALLOCATE(hamiltonian_system_errors, batch)
  ALLOCATE(hamiltonian_device_error, 1u)
  ALLOCATE(integral_system_errors, batch)
  ALLOCATE(integral_device_error, 1u)
  ALLOCATE(coordination_system_errors, batch)
  ALLOCATE(coordination_device_error, 1u)
  ALLOCATE(classical_system_errors, batch)
  ALLOCATE(classical_device_error, 1u)
  ALLOCATE(external_system_errors, batch)
  ALLOCATE(external_device_error, 1u)
  ALLOCATE(composition_system_errors, batch)
  ALLOCATE(composition_plan_error, 1u)
#undef ALLOCATE
  if (status != cudaSuccess) {
    return status;
  }

  std::vector<std::uint8_t> all_requested(batch, 1u);
  std::vector<gpuxtb_status_t> all_success(batch, GPUXTB_STATUS_SUCCESS);
  std::vector<std::uint8_t> all_converged(batch, 1u);
  std::vector<std::uint64_t> generations(batch, kGeometryGeneration);
  /* Deliberately stale last-mixed potentials. The execution must rebuild
   * Hamiltonian force inputs from the final raw SCC multipoles below. */
  std::vector<double> last_mixed_shell_scalar(shells);
  std::vector<double> last_mixed_dipole(coordinates);
  std::vector<double> last_mixed_quadrupole(atoms * 6u);
  for (std::size_t index = 0; index < last_mixed_shell_scalar.size(); ++index) {
    last_mixed_shell_scalar[index] = -0.73 - 0.01 * static_cast<double>(index);
  }
  for (std::size_t index = 0; index < last_mixed_dipole.size(); ++index) {
    last_mixed_dipole[index] = 0.41 + 0.002 * static_cast<double>(index);
  }
  for (std::size_t index = 0; index < last_mixed_quadrupole.size(); ++index) {
    last_mixed_quadrupole[index] = -0.29 + 0.001 * static_cast<double>(index);
  }
#define COPY(field, source)                                               \
  if (status == cudaSuccess) {                                            \
    status = d.field.copy_from((source).data(), (source).size(), stream); \
  }
  COPY(positions, h.positions)
  COPY(point_positions, h.point_positions)
  COPY(point_charges, h.point_charges)
  COPY(point_hardnesses, h.point_hardnesses)
  COPY(coordination, h.coordination)
  COPY(overlap, h.overlap)
  COPY(density, h.density)
  COPY(weighted_density, h.weighted_density)
  COPY(shell_scalar, last_mixed_shell_scalar)
  COPY(dipole_potential, last_mixed_dipole)
  COPY(quadrupole_potential, last_mixed_quadrupole)
  COPY(shell_charges, h.shell_charges)
  COPY(atomic_charges, h.atomic_charges)
  COPY(atomic_dipoles, h.atomic_dipoles)
  COPY(atomic_quadrupoles, h.atomic_quadrupoles)
  COPY(scc_free_energy, h.scc_free_energy)
  COPY(repulsion_energy, h.repulsion_energy)
  COPY(es2_matrix, h.es2_matrix)
  COPY(aes2_pairs, h.aes2_pairs)
  COPY(requested, all_requested)
  COPY(statuses, all_success)
  COPY(converged, all_converged)
  COPY(geometry_generations, generations)
#undef COPY
  if (status != cudaSuccess) {
    return status;
  }

  const Gfn2IntegralDeviceBatch integral_batch{
      h.basis.batch_size,
      h.basis.total_atoms,
      h.basis.total_shells,
      h.basis.total_orbitals,
      h.basis.total_primitives,
      h.integrals.total_matrix_elements,
      h.h0.shell_pair_offsets.back(),
      h.maximum_system_shells,
      h.integrals.integral_cutoff,
      kPlanToken,
      static_cast<std::int64_t>(h.basis.atom_offsets.size()),
      static_cast<std::int64_t>(h.basis.batch_shell_offsets.size()),
      static_cast<std::int64_t>(h.basis.batch_orbital_offsets.size()),
      static_cast<std::int64_t>(h.integrals.matrix_offsets.size()),
      static_cast<std::int64_t>(h.h0.shell_pair_offsets.size()),
      static_cast<std::int64_t>(h.basis.atom_shell_offsets.size()),
      static_cast<std::int64_t>(h.basis.shell_orbital_offsets.size()),
      static_cast<std::int64_t>(h.basis.shell_primitive_offsets.size()),
      static_cast<std::int64_t>(h.basis.shell_to_atom.size()),
      static_cast<std::int64_t>(h.basis.angular_momenta.size()),
      static_cast<std::int64_t>(h.basis.primitive_exponents.size()),
      static_cast<std::int64_t>(h.basis.primitive_coefficients.size()),
      d.atom_offsets.get(),
      d.batch_shell_offsets.get(),
      d.batch_orbital_offsets.get(),
      d.integral_matrix_offsets.get(),
      d.shell_pair_offsets.get(),
      d.atom_shell_offsets.get(),
      d.shell_orbital_offsets.get(),
      d.shell_primitive_offsets.get(),
      d.shell_to_atom.get(),
      d.angular_momenta.get(),
      d.primitive_exponents.get(),
      d.primitive_coefficients.get()};
  const Gfn2H0DevicePlan h0_plan{
      h.basis.total_atoms,      h.basis.total_shells,           h.basis.total_shells,
      h.basis.total_shells,     h.h0.shell_pair_offsets.back(), kPlanToken,
      d.atomic_radii.get(),     d.shell_levels.get(),           d.shell_coordination_scale.get(),
      d.shell_polynomial.get(), d.shell_pair_scale.get()};
  const Gfn2HamiltonianDeviceBatch hamiltonian_batch{
      h.basis.batch_size,
      h.basis.total_atoms,
      h.basis.total_shells,
      h.basis.total_orbitals,
      h.integrals.total_matrix_elements,
      kPlanToken,
      static_cast<std::int64_t>(h.basis.atom_offsets.size()),
      static_cast<std::int64_t>(h.basis.batch_shell_offsets.size()),
      static_cast<std::int64_t>(h.basis.batch_orbital_offsets.size()),
      static_cast<std::int64_t>(h.integrals.matrix_offsets.size()),
      static_cast<std::int64_t>(h.basis.atom_shell_offsets.size()),
      static_cast<std::int64_t>(h.basis.shell_orbital_offsets.size()),
      static_cast<std::int64_t>(h.basis.shell_to_atom.size()),
      static_cast<std::int64_t>(h.orbital_to_shell.size()),
      static_cast<std::int64_t>(h.orbital_to_atom.size()),
      d.atom_offsets.get(),
      d.batch_shell_offsets.get(),
      d.batch_orbital_offsets.get(),
      d.integral_matrix_offsets.get(),
      d.atom_shell_offsets.get(),
      d.shell_orbital_offsets.get(),
      d.shell_to_atom.get(),
      d.orbital_to_shell.get(),
      d.orbital_to_atom.get()};

  const Gfn2GeometryDeviceBatch geometry_batch{static_cast<std::int64_t>(batch),
                                               static_cast<std::int64_t>(atoms),
                                               static_cast<std::int64_t>(pairs),
                                               static_cast<std::int64_t>(batch + 1u),
                                               static_cast<std::int64_t>(batch + 1u),
                                               static_cast<std::int64_t>(atoms),
                                               static_cast<std::int64_t>(coordinates),
                                               kPlanToken,
                                               d.atom_offsets.get(),
                                               d.pair_offsets.get(),
                                               d.covalent_radii.get()};
  const Gfn2GeometryDeviceCache geometry_cache{
      d.geometry_pairs.get(),
      static_cast<std::int64_t>(pairs * kGfn2GeometryPairDataElements),
      d.geometry_coordination.get(),
      static_cast<std::int64_t>(atoms),
      d.geometry_generations.get(),
      static_cast<std::int64_t>(batch),
      kPlanToken};
  const Gfn2GeometryDeviceWorkspace geometry_update_workspace{
      d.geometry_pair_scratch.get(),
      static_cast<std::int64_t>(pairs * kGfn2GeometryPairDataElements),
      d.geometry_coordination_scratch.get(),
      static_cast<std::int64_t>(atoms),
      d.geometry_update_gradient_scratch.get(),
      static_cast<std::int64_t>(coordinates),
      d.geometry_update_sequence.get(),
      1,
      kPlanToken};

  const Gfn2ES2DeviceBatch es2_batch{static_cast<std::int64_t>(batch),
                                     static_cast<std::int64_t>(atoms),
                                     static_cast<std::int64_t>(shells),
                                     h.es2_plan.total_matrix_elements(),
                                     kPlanToken,
                                     static_cast<std::int64_t>(batch + 1u),
                                     static_cast<std::int64_t>(batch + 1u),
                                     static_cast<std::int64_t>(atoms + 1u),
                                     static_cast<std::int64_t>(batch + 1u),
                                     static_cast<std::int64_t>(shells),
                                     static_cast<std::int64_t>(shells),
                                     d.atom_offsets.get(),
                                     d.batch_shell_offsets.get(),
                                     d.atom_shell_offsets.get(),
                                     d.es2_matrix_offsets.get(),
                                     d.shell_to_atom.get(),
                                     d.es2_hardness.get()};
  const Gfn2ES2DeviceCache es2_cache{d.es2_matrix.get(), h.es2_plan.total_matrix_elements(),
                                     kGeometryGeneration, kPlanToken};
  const Gfn2ES2DeviceWorkspace es2_update_workspace{d.es2_matrix_scratch.get(),
                                                    h.es2_plan.total_matrix_elements(),
                                                    nullptr,
                                                    0,
                                                    nullptr,
                                                    0,
                                                    nullptr,
                                                    0};
  const Gfn2AES2DeviceBatch aes2_batch{static_cast<std::int64_t>(batch),
                                       static_cast<std::int64_t>(atoms),
                                       static_cast<std::int64_t>(pairs),
                                       kPlanToken,
                                       static_cast<std::int64_t>(batch + 1u),
                                       static_cast<std::int64_t>(batch + 1u),
                                       static_cast<std::int64_t>(atoms),
                                       static_cast<std::int64_t>(atoms),
                                       static_cast<std::int64_t>(atoms),
                                       static_cast<std::int64_t>(atoms),
                                       d.atom_offsets.get(),
                                       d.pair_offsets.get(),
                                       d.aes2_dipole_kernel.get(),
                                       d.aes2_quadrupole_kernel.get(),
                                       d.aes2_radius.get(),
                                       d.aes2_valence_cn.get()};
  const Gfn2AES2DeviceCache aes2_cache{d.aes2_pairs.get(),
                                       static_cast<std::int64_t>(h.aes2_pairs.size()),
                                       kGeometryGeneration, kPlanToken};
  const Gfn2AES2DeviceWorkspace aes2_update_workspace{
      d.aes2_pair_scratch.get(),
      static_cast<std::int64_t>(h.aes2_pairs.size()),
      nullptr,
      0,
      nullptr,
      0,
      nullptr,
      0,
      nullptr,
      0,
      d.aes2_peer_error_scratch.get(),
      1};
  const Gfn2ExternalPointChargeDeviceBatch external_batch{static_cast<std::int64_t>(batch),
                                                          static_cast<std::int64_t>(atoms),
                                                          static_cast<std::int64_t>(shells),
                                                          static_cast<std::int64_t>(batch),
                                                          d.atom_offsets.get(),
                                                          d.batch_shell_offsets.get(),
                                                          d.point_offsets.get(),
                                                          d.shell_to_atom.get(),
                                                          d.external_shell_hardness.get(),
                                                          d.positions.get(),
                                                          d.point_positions.get(),
                                                          d.point_charges.get(),
                                                          d.point_hardnesses.get(),
                                                          kPlanToken};

  const std::uint32_t classical_components =
      static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kRepulsion) |
      static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kES2) |
      static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kAES2);
  const Gfn2ClassicalForceDevicePlan classical_plan{static_cast<std::int64_t>(batch),
                                                    static_cast<std::int64_t>(atoms),
                                                    static_cast<std::int64_t>(shells),
                                                    classical_components,
                                                    kGeometryGeneration,
                                                    kPlanToken,
                                                    d.atom_offsets.get(),
                                                    d.atomic_numbers.get(),
                                                    geometry_batch,
                                                    geometry_cache,
                                                    es2_batch,
                                                    es2_cache,
                                                    aes2_batch,
                                                    aes2_cache,
                                                    {},
                                                    {},
                                                    {}};
  const std::uint32_t composition_components =
      static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kElectronicGradient) |
      static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kClassicalForce) |
      static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kExplicitPointChargeForce);
  const Gfn2ForceCompositionDeviceBatch composition_batch{static_cast<std::int64_t>(batch),
                                                          static_cast<std::int64_t>(atoms),
                                                          static_cast<std::int64_t>(batch),
                                                          static_cast<std::int64_t>(batch + 1u),
                                                          static_cast<std::int64_t>(batch + 1u),
                                                          d.atom_offsets.get(),
                                                          d.point_offsets.get(),
                                                          composition_components,
                                                          kPlanToken};
  constexpr std::uint32_t scc_components =
      static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES2) |
      static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES3) |
      static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kAES2) |
      static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kExplicitPointCharge);
  Gfn2PostSccPotentialDevicePlan post_scc_plan{};
  post_scc_plan.enabled_components = scc_components;
  post_scc_plan.geometry_generation = kGeometryGeneration;
  post_scc_plan.plan_token = kPlanToken;
  auto& potential_batch = post_scc_plan.potential_batch;
  potential_batch.batch_size = static_cast<std::int64_t>(batch);
  potential_batch.total_atoms = static_cast<std::int64_t>(atoms);
  potential_batch.total_shells = static_cast<std::int64_t>(shells);
  potential_batch.plan_token = kPlanToken;
  potential_batch.atom_offset_count = static_cast<std::int64_t>(batch + 1u);
  potential_batch.batch_shell_offset_count = static_cast<std::int64_t>(batch + 1u);
  potential_batch.qsh_offset_count = static_cast<std::int64_t>(batch + 1u);
  potential_batch.qat_offset_count = static_cast<std::int64_t>(batch + 1u);
  potential_batch.dipole_offset_count = static_cast<std::int64_t>(batch + 1u);
  potential_batch.quadrupole_offset_count = static_cast<std::int64_t>(batch + 1u);
  potential_batch.shell_to_atom_count = static_cast<std::int64_t>(shells);
  potential_batch.atom_offsets = d.atom_offsets.get();
  potential_batch.batch_shell_offsets = d.batch_shell_offsets.get();
  potential_batch.qsh_offsets = d.qsh_offsets.get();
  potential_batch.qat_offsets = d.atom_offsets.get();
  potential_batch.dipole_offsets = d.dipole_offsets.get();
  potential_batch.quadrupole_offsets = d.quadrupole_offsets.get();
  potential_batch.shell_to_atom = d.shell_to_atom.get();

  auto& bridge = post_scc_plan.scalar_bridge_batch;
  bridge.topology.memory_space = gpuxtb::detail::Gfn2PlanMemorySpace::kCudaDevice;
  bridge.topology.pair_map_kind = gpuxtb::detail::Gfn2PairMapKind::kNone;
  bridge.topology.plan_token = kPlanToken;
  bridge.topology.batch_size = static_cast<std::int64_t>(batch);
  bridge.topology.total_atoms = static_cast<std::int64_t>(atoms);
  bridge.topology.total_shells = static_cast<std::int64_t>(shells);
  bridge.topology.atom_offset_count = static_cast<std::int64_t>(batch + 1u);
  bridge.topology.batch_shell_offset_count = static_cast<std::int64_t>(batch + 1u);
  bridge.topology.shell_to_atom_count = static_cast<std::int64_t>(shells);
  bridge.topology.atom_offsets = d.atom_offsets.get();
  bridge.topology.batch_shell_offsets = d.batch_shell_offsets.get();
  bridge.topology.shell_to_atom = d.shell_to_atom.get();
  bridge.qsh_offset_count = static_cast<std::int64_t>(batch + 1u);
  bridge.qat_offset_count = static_cast<std::int64_t>(batch + 1u);
  bridge.qsh_offsets = d.qsh_offsets.get();
  bridge.qat_offsets = d.atom_offsets.get();

  post_scc_plan.es2_batch = es2_batch;
  post_scc_plan.es2_cache = es2_cache;
  post_scc_plan.es3_batch = {static_cast<std::int64_t>(batch),
                             static_cast<std::int64_t>(shells),
                             static_cast<std::int64_t>(batch + 1u),
                             static_cast<std::int64_t>(shells),
                             d.batch_shell_offsets.get(),
                             d.es3_gamma.get(),
                             kPlanToken};
  post_scc_plan.aes2_batch = aes2_batch;
  post_scc_plan.aes2_cache = aes2_cache;
  post_scc_plan.external_point_charge_batch = external_batch;
  post_scc_plan.external_point_charge_cache = {d.external_shell_potential.get(),
                                               static_cast<std::int64_t>(shells),
                                               kGeometryGeneration, kPlanToken};

  d.plan = {};
  d.plan.compute_forces = 1u;
  d.plan.scc_potential_components = scc_components;
  d.plan.scc_energy_components = scc_components;
  d.plan.plan_token = kPlanToken;
  d.plan.total_energy_batch = {static_cast<std::int64_t>(batch), 0u, kPlanToken};
  d.plan.post_scc_potential_plan = post_scc_plan;
  d.plan.integral_batch = integral_batch;
  d.plan.h0_plan = h0_plan;
  d.plan.hamiltonian_batch = hamiltonian_batch;
  d.plan.coordination_batch = geometry_batch;
  d.plan.coordination_cache = geometry_cache;
  d.plan.geometry_generation = kGeometryGeneration;
  d.plan.classical_plan = classical_plan;
  d.plan.external_point_charge_batch = external_batch;
  d.plan.force_composition_batch = composition_batch;

  d.input = {};
  d.input.total_energy = {d.scc_free_energy.get(),
                          static_cast<std::int64_t>(batch),
                          d.repulsion_energy.get(),
                          static_cast<std::int64_t>(batch),
                          nullptr,
                          0,
                          kPlanToken};
  d.input.scc_state = {d.statuses.get(), d.converged.get(), static_cast<std::int64_t>(batch),
                       kPlanToken};
  d.input.force_activity = {d.requested.get(), d.statuses.get(), static_cast<std::int64_t>(batch),
                            kPlanToken};
  d.input.post_scc_potential = {
      {d.requested.get(), d.statuses.get(), static_cast<std::int64_t>(batch), kPlanToken},
      d.shell_charges.get(),
      static_cast<std::int64_t>(shells),
      d.atomic_charges.get(),
      static_cast<std::int64_t>(atoms),
      d.atomic_dipoles.get(),
      static_cast<std::int64_t>(coordinates),
      d.atomic_quadrupoles.get(),
      static_cast<std::int64_t>(atoms * 6u),
      kPlanToken};
  d.input.h0 = {d.positions.get(),
                static_cast<std::int64_t>(coordinates),
                d.geometry_coordination.get(),
                static_cast<std::int64_t>(atoms),
                d.overlap.get(),
                static_cast<std::int64_t>(matrices),
                d.density.get(),
                static_cast<std::int64_t>(matrices),
                d.weighted_density.get(),
                static_cast<std::int64_t>(matrices),
                kPlanToken};
  /* These are exact aliases of the fresh post-SCC publication targets. */
  d.input.hamiltonian = {d.density.get(),
                         static_cast<std::int64_t>(matrices),
                         d.shell_scalar.get(),
                         static_cast<std::int64_t>(shells),
                         d.dipole_potential.get(),
                         static_cast<std::int64_t>(coordinates),
                         d.quadrupole_potential.get(),
                         static_cast<std::int64_t>(atoms * 6u),
                         kPlanToken};
  d.input.classical = {d.positions.get(),
                       static_cast<std::int64_t>(coordinates),
                       d.geometry_coordination.get(),
                       static_cast<std::int64_t>(atoms),
                       d.shell_charges.get(),
                       static_cast<std::int64_t>(shells),
                       d.atomic_charges.get(),
                       static_cast<std::int64_t>(atoms),
                       d.atomic_dipoles.get(),
                       static_cast<std::int64_t>(coordinates),
                       d.atomic_quadrupoles.get(),
                       static_cast<std::int64_t>(atoms * 6u),
                       kPlanToken};
  d.input.external_shell_charges = d.shell_charges.get();
  d.input.external_shell_elements = static_cast<std::int64_t>(shells);
  d.input.plan_token = kPlanToken;

  d.results = {};
  d.results.energy = {d.public_energy.get(), static_cast<std::int64_t>(batch), kPlanToken};
  d.results.forces = {d.public_qm_force.get(), static_cast<std::int64_t>(coordinates),
                      d.public_point_force.get(), static_cast<std::int64_t>(point_coordinates),
                      kPlanToken};
  d.results.plan_token = kPlanToken;

  d.intermediates = {};
  d.intermediates.energy = {d.staged_energy.get(), static_cast<std::int64_t>(batch), kPlanToken};
  d.intermediates.post_scc_potential = {
      {d.post_complete_shell.get(), static_cast<std::int64_t>(shells), d.post_complete_atomic.get(),
       static_cast<std::int64_t>(atoms), d.dipole_potential.get(),
       static_cast<std::int64_t>(coordinates), d.quadrupole_potential.get(),
       static_cast<std::int64_t>(atoms * 6u), kPlanToken},
      d.shell_scalar.get(),
      static_cast<std::int64_t>(shells),
      kPlanToken};
  auto& post_intermediates = d.intermediates.post_scc_potential_intermediates;
  post_intermediates.es2_shell = d.post_es2_shell.get();
  post_intermediates.es2_shell_elements = static_cast<std::int64_t>(shells);
  post_intermediates.es3_shell = d.post_es3_shell.get();
  post_intermediates.es3_shell_elements = static_cast<std::int64_t>(shells);
  post_intermediates.aes2_atomic = d.post_aes2_atomic.get();
  post_intermediates.aes2_atomic_elements = static_cast<std::int64_t>(atoms);
  post_intermediates.aes2_dipole = d.post_aes2_dipole.get();
  post_intermediates.aes2_dipole_elements = static_cast<std::int64_t>(coordinates);
  post_intermediates.aes2_quadrupole = d.post_aes2_quadrupole.get();
  post_intermediates.aes2_quadrupole_elements = static_cast<std::int64_t>(atoms * 6u);
  post_intermediates.complete = {d.post_staged_shell.get(),
                                 static_cast<std::int64_t>(shells),
                                 d.post_staged_atomic.get(),
                                 static_cast<std::int64_t>(atoms),
                                 d.post_staged_dipole.get(),
                                 static_cast<std::int64_t>(coordinates),
                                 d.post_staged_quadrupole.get(),
                                 static_cast<std::int64_t>(atoms * 6u),
                                 kPlanToken};
  post_intermediates.shell_scalar = d.post_staged_shell_scalar.get();
  post_intermediates.shell_scalar_elements = static_cast<std::int64_t>(shells);
  post_intermediates.plan_token = kPlanToken;
  d.intermediates.h0 = {d.overlap_adjoint.get(),
                        static_cast<std::int64_t>(matrices),
                        d.coordination_adjoint.get(),
                        static_cast<std::int64_t>(atoms),
                        d.electronic_gradient.get(),
                        static_cast<std::int64_t>(coordinates),
                        kPlanToken};
  d.intermediates.hamiltonian = {d.overlap_adjoint.get(),
                                 static_cast<std::int64_t>(matrices),
                                 d.dipole_adjoint.get(),
                                 static_cast<std::int64_t>(3u * matrices),
                                 d.quadrupole_adjoint.get(),
                                 static_cast<std::int64_t>(6u * matrices),
                                 kPlanToken};
  d.intermediates.classical = {d.classical_force.get(), static_cast<std::int64_t>(coordinates),
                               kPlanToken};
  d.intermediates.explicit_qm_forces = d.explicit_qm_force.get();
  d.intermediates.explicit_qm_force_elements = static_cast<std::int64_t>(coordinates);
  d.intermediates.explicit_point_forces = d.explicit_point_force.get();
  d.intermediates.explicit_point_force_elements = static_cast<std::int64_t>(point_coordinates);
  d.intermediates.forces = {d.staged_qm_force.get(), static_cast<std::int64_t>(coordinates),
                            d.staged_point_force.get(),
                            static_cast<std::int64_t>(point_coordinates), kPlanToken};
  d.intermediates.plan_token = kPlanToken;

  const Gfn2AES2DeviceWorkspace classical_aes2_workspace{
      nullptr,
      0,
      nullptr,
      0,
      nullptr,
      0,
      d.classical_aes2_gradient_scratch.get(),
      static_cast<std::int64_t>(coordinates),
      d.classical_aes2_coordination_scratch.get(),
      static_cast<std::int64_t>(atoms),
      nullptr,
      0};
  const Gfn2GeometryDeviceWorkspace classical_geometry_workspace{
      nullptr,
      0,
      nullptr,
      0,
      d.classical_geometry_gradient_scratch.get(),
      static_cast<std::int64_t>(coordinates),
      d.classical_primitive_sequence.get(),
      1,
      kPlanToken};
  d.workspace = {};
  d.workspace.total_energy = {d.total_sequence.get(), 1, kPlanToken};
  auto& post_workspace = d.workspace.post_scc_potential;
  post_workspace.es2.shell_scratch = d.post_es2_shell_scratch.get();
  post_workspace.es2.shell_elements = static_cast<std::int64_t>(shells);
  post_workspace.aes2.potential_scratch = d.post_aes2_potential_scratch.get();
  post_workspace.aes2.potential_elements = h.aes2_plan.potential_scratch_elements();
  post_workspace.aes2.scc_peer_error_scratch = d.post_aes2_peer_scratch.get();
  post_workspace.aes2.scc_peer_error_elements = 1;
  post_workspace.composition.shell_scratch = d.post_compose_shell_scratch.get();
  post_workspace.composition.shell_elements = static_cast<std::int64_t>(shells);
  post_workspace.composition.atom_scratch = d.post_compose_atomic_scratch.get();
  post_workspace.composition.atom_elements = static_cast<std::int64_t>(atoms);
  post_workspace.composition.dipole_scratch = d.post_compose_dipole_scratch.get();
  post_workspace.composition.dipole_elements = static_cast<std::int64_t>(coordinates);
  post_workspace.composition.quadrupole_scratch = d.post_compose_quadrupole_scratch.get();
  post_workspace.composition.quadrupole_elements = static_cast<std::int64_t>(atoms * 6u);
  post_workspace.composition.sequence_active = d.post_compose_sequence.get();
  post_workspace.composition.sequence_elements = 1;
  post_workspace.composition.plan_token = kPlanToken;
  post_workspace.scalar_bridge.shell_scratch = d.post_bridge_scratch.get();
  post_workspace.scalar_bridge.shell_elements = static_cast<std::int64_t>(shells);
  post_workspace.scalar_bridge.sequence_active = d.post_bridge_sequence.get();
  post_workspace.scalar_bridge.sequence_elements = 1;
  post_workspace.scalar_bridge.plan_token = kPlanToken;
  post_workspace.active_mask = d.post_active.get();
  post_workspace.active_elements = static_cast<std::int64_t>(batch);
  post_workspace.sequence_active = d.post_sequence.get();
  post_workspace.sequence_elements = 1;
  post_workspace.stage_system_errors = d.post_stage_system_errors.get();
  post_workspace.stage_system_error_elements = static_cast<std::int64_t>(batch);
  post_workspace.stage_device_error = d.post_stage_device_error.get();
  post_workspace.stage_device_error_elements = 1;
  post_workspace.plan_token = kPlanToken;
  d.workspace.h0 = {d.h0_overlap_scratch.get(),
                    static_cast<std::int64_t>(matrices),
                    d.h0_coordination_scratch.get(),
                    static_cast<std::int64_t>(atoms),
                    d.h0_gradient_scratch.get(),
                    static_cast<std::int64_t>(coordinates),
                    d.h0_sequence.get(),
                    1,
                    kPlanToken};
  d.workspace.hamiltonian = {d.hamiltonian_overlap_scratch.get(),
                             static_cast<std::int64_t>(matrices),
                             d.hamiltonian_dipole_scratch.get(),
                             static_cast<std::int64_t>(3u * matrices),
                             d.hamiltonian_quadrupole_scratch.get(),
                             static_cast<std::int64_t>(6u * matrices),
                             d.hamiltonian_sequence.get(),
                             1,
                             kPlanToken};
  d.workspace.integral = {d.integral_gradient_scratch.get(), static_cast<std::int64_t>(coordinates),
                          d.integral_sequence.get(), 1, kPlanToken};
  d.workspace.electronic = {d.h0_success.get(), d.hamiltonian_success.get(),
                            static_cast<std::int64_t>(batch), kPlanToken};
  d.workspace.coordination = {nullptr,
                              0,
                              nullptr,
                              0,
                              d.coordination_gradient_scratch.get(),
                              static_cast<std::int64_t>(coordinates),
                              d.coordination_sequence.get(),
                              1,
                              kPlanToken};
  d.workspace.classical = {d.classical_gradient_scratch.get(),
                           static_cast<std::int64_t>(coordinates),
                           d.classical_force_scratch.get(),
                           static_cast<std::int64_t>(coordinates),
                           d.classical_coordination_adjoint.get(),
                           static_cast<std::int64_t>(atoms),
                           d.classical_selected.get(),
                           static_cast<std::int64_t>(batch),
                           d.classical_primitive_system_errors.get(),
                           static_cast<std::int64_t>(batch),
                           d.classical_primitive_device_error.get(),
                           1,
                           d.classical_sequence.get(),
                           1,
                           classical_aes2_workspace,
                           {},
                           classical_geometry_workspace,
                           kPlanToken};
  d.workspace.external_point_charge = {d.external_qm_scratch.get(),
                                       static_cast<std::int64_t>(coordinates),
                                       d.external_point_scratch.get(),
                                       static_cast<std::int64_t>(point_coordinates),
                                       d.external_sequence.get(),
                                       1,
                                       kPlanToken};
  d.workspace.force_composition = {d.composition_qm_scratch.get(),
                                   static_cast<std::int64_t>(coordinates),
                                   d.composition_point_scratch.get(),
                                   static_cast<std::int64_t>(point_coordinates),
                                   d.composition_sequence.get(),
                                   1,
                                   kPlanToken};
  d.workspace.energy_success_mask = d.energy_success.get();
  d.workspace.post_scc_success_mask = d.post_scc_success.get();
  d.workspace.electronic_success_mask = d.electronic_success.get();
  d.workspace.coordination_success_mask = d.coordination_success.get();
  d.workspace.classical_success_mask = d.classical_success.get();
  d.workspace.external_success_mask = d.external_success.get();
  d.workspace.mask_elements = static_cast<std::int64_t>(batch);
  d.workspace.plan_failure = d.plan_failure.get();
  d.workspace.plan_failure_elements = 1;
  d.workspace.plan_token = kPlanToken;

  d.diagnostics = {};
  d.diagnostics.execution_system_errors = d.execution_system_errors.get();
  d.diagnostics.execution_device_error = d.execution_device_error.get();
  d.diagnostics.total_energy_system_errors = d.total_system_errors.get();
  d.diagnostics.total_energy_device_error = d.total_device_error.get();
  d.diagnostics.post_scc_potential = {d.post_system_errors.get(), d.post_device_error.get(),
                                      static_cast<std::int64_t>(batch), kPlanToken};
  d.diagnostics.electronic = {d.h0_system_errors.get(),          d.h0_device_error.get(),
                              d.hamiltonian_system_errors.get(), d.hamiltonian_device_error.get(),
                              d.integral_system_errors.get(),    d.integral_device_error.get(),
                              static_cast<std::int64_t>(batch),  kPlanToken};
  d.diagnostics.coordination_system_errors = d.coordination_system_errors.get();
  d.diagnostics.coordination_device_error = d.coordination_device_error.get();
  d.diagnostics.classical_system_errors = d.classical_system_errors.get();
  d.diagnostics.classical_device_error = d.classical_device_error.get();
  d.diagnostics.external_system_errors = d.external_system_errors.get();
  d.diagnostics.external_device_error = d.external_device_error.get();
  d.diagnostics.force_composition_system_errors = d.composition_system_errors.get();
  d.diagnostics.force_composition_plan_error = d.composition_plan_error.get();
  d.diagnostics.batch_elements = static_cast<std::int64_t>(batch);
  d.diagnostics.plan_token = kPlanToken;

  status = reset_gfn2_geometry_device_errors_cuda(static_cast<std::int64_t>(batch),
                                                  d.coordination_system_errors.get(),
                                                  d.coordination_device_error.get(), stream);
  int device_id = -1;
  std::string parameter_error;
  if (status == cudaSuccess &&
      (cudaGetDevice(&device_id) != cudaSuccess ||
       !gpuxtb::detail::ensure_cuda_gfn2_parameters(device_id, parameter_error))) {
    return cudaErrorInvalidValue;
  }
  if (status == cudaSuccess) {
    status = update_gfn2_geometry_cache_cuda(geometry_batch, d.positions.get(), kGeometryGeneration,
                                             geometry_cache, geometry_update_workspace,
                                             d.coordination_system_errors.get(),
                                             d.coordination_device_error.get(), stream);
  }
  if (status == cudaSuccess) {
    status = reset_gfn2_es2_device_error_cuda(d.classical_primitive_device_error.get(), stream);
  }
  if (status == cudaSuccess) {
    status = update_gfn2_es2_geometry_cache_cuda(es2_batch, d.positions.get(), es2_cache,
                                                 es2_update_workspace,
                                                 d.classical_primitive_device_error.get(), stream);
  }
  if (status == cudaSuccess) {
    status = reset_gfn2_aes2_device_errors_cuda(static_cast<std::int64_t>(batch),
                                                d.classical_primitive_system_errors.get(),
                                                d.classical_primitive_device_error.get(), stream);
  }
  if (status == cudaSuccess) {
    status = update_gfn2_aes2_geometry_cache_cuda(
        aes2_batch, d.positions.get(), d.geometry_coordination.get(), aes2_cache,
        aes2_update_workspace, d.classical_primitive_system_errors.get(),
        d.classical_primitive_device_error.get(), stream);
  }
  return status;
}

cudaError_t seed_public_results(DeviceFixture& d, const HostCase& h, cudaStream_t stream) {
  const std::vector<double> energy(h.batch_size, kSentinel);
  const std::vector<double> qm(static_cast<std::size_t>(h.basis.total_atoms) * 3u, kSentinel);
  const std::vector<double> point(h.point_positions.size(), kSentinel);
  cudaError_t status = d.public_energy.copy_from(energy.data(), energy.size(), stream);
  if (status == cudaSuccess) {
    status = d.public_qm_force.copy_from(qm.data(), qm.size(), stream);
  }
  return status == cudaSuccess ? d.public_point_force.copy_from(point.data(), point.size(), stream)
                               : status;
}

cudaError_t upload_raw_state(DeviceFixture& d, const HostCase& h, cudaStream_t stream) {
  cudaError_t status =
      d.shell_charges.copy_from(h.shell_charges.data(), h.shell_charges.size(), stream);
  if (status == cudaSuccess) {
    status = d.atomic_charges.copy_from(h.atomic_charges.data(), h.atomic_charges.size(), stream);
  }
  if (status == cudaSuccess) {
    status = d.atomic_dipoles.copy_from(h.atomic_dipoles.data(), h.atomic_dipoles.size(), stream);
  }
  if (status == cudaSuccess) {
    status = d.atomic_quadrupoles.copy_from(h.atomic_quadrupoles.data(),
                                            h.atomic_quadrupoles.size(), stream);
  }
  if (status == cudaSuccess) {
    status =
        d.scc_free_energy.copy_from(h.scc_free_energy.data(), h.scc_free_energy.size(), stream);
  }
  return status == cudaSuccess ? d.repulsion_energy.copy_from(h.repulsion_energy.data(),
                                                              h.repulsion_energy.size(), stream)
                               : status;
}

cudaError_t launch_execution(DeviceFixture& d, cudaStream_t stream) {
  return execute_gfn2_energy_force_cuda(d.plan, d.input, d.results, d.intermediates, d.workspace,
                                        d.diagnostics, stream);
}

int compare_success(DeviceFixture& d, const HostCase& h, cudaStream_t stream) {
  std::vector<double> energy(h.batch_size);
  std::vector<double> qm(h.expected_qm_force.size());
  std::vector<double> point(h.expected_point_force.size());
  std::vector<std::uint32_t> errors(h.batch_size);
  std::vector<std::uint32_t> classical_errors(h.batch_size);
  std::vector<std::uint32_t> primitive_errors(h.batch_size);
  std::uint32_t device_error = 1u;
  std::uint32_t classical_device_error = 0u;
  std::uint32_t primitive_device_error = 0u;
  CUDA_CHECK(d.public_energy.copy_to(energy.data(), energy.size(), stream));
  CUDA_CHECK(d.public_qm_force.copy_to(qm.data(), qm.size(), stream));
  CUDA_CHECK(d.public_point_force.copy_to(point.data(), point.size(), stream));
  CUDA_CHECK(d.execution_system_errors.copy_to(errors.data(), errors.size(), stream));
  CUDA_CHECK(d.execution_device_error.copy_to(&device_error, 1u, stream));
  CUDA_CHECK(
      d.classical_system_errors.copy_to(classical_errors.data(), classical_errors.size(), stream));
  CUDA_CHECK(d.classical_device_error.copy_to(&classical_device_error, 1u, stream));
  CUDA_CHECK(d.classical_primitive_system_errors.copy_to(primitive_errors.data(),
                                                         primitive_errors.size(), stream));
  CUDA_CHECK(d.classical_primitive_device_error.copy_to(&primitive_device_error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (device_error != 0u) {
    std::fprintf(stderr,
                 "execution device error: %u; classical device error: %u; primitive device "
                 "error: %u\n",
                 device_error, classical_device_error, primitive_device_error);
    for (std::size_t system = 0; system < classical_errors.size(); ++system) {
      std::fprintf(stderr, "classical[%zu]=%u primitive[%zu]=%u execution[%zu]=%u\n", system,
                   classical_errors[system], system, primitive_errors[system], system,
                   errors[system]);
    }
  }
  CHECK(device_error == 0u);
  CHECK(std::all_of(errors.begin(), errors.end(), [](std::uint32_t value) { return value == 0u; }));
  for (std::size_t system = 0; system < energy.size(); ++system) {
    CHECK(near(energy[system], h.expected_energy[system]));
  }
  for (std::size_t coordinate = 0; coordinate < qm.size(); ++coordinate) {
    CHECK(near(qm[coordinate], h.expected_qm_force[coordinate], 2.0e-8, 2.0e-8));
  }
  for (std::size_t coordinate = 0; coordinate < point.size(); ++coordinate) {
    CHECK(near(point[coordinate], h.expected_point_force[coordinate], 2.0e-9, 2.0e-9));
  }
  return 0;
}

int test_batch_parity(std::size_t batch_size) {
  HostCase host;
  std::string error;
  CHECK(make_case(batch_size, host, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(initialize_device(device, host, stream));
  CUDA_CHECK(seed_public_results(device, host, stream));
  CUDA_CHECK(launch_execution(device, stream));
  if (const int line = compare_success(device, host, stream); line != 0) {
    return line;
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_energy_only_ignores_force_bindings() {
  HostCase host;
  std::string error;
  CHECK(make_case(8u, host, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(initialize_device(device, host, stream));
  CUDA_CHECK(seed_public_results(device, host, stream));

  Gfn2EnergyForceExecutionDevicePlan plan{};
  plan.compute_forces = 0u;
  plan.plan_token = kPlanToken;
  plan.total_energy_batch = device.plan.total_energy_batch;
  Gfn2EnergyForceExecutionDeviceInput input{};
  input.total_energy = device.input.total_energy;
  input.scc_state = device.input.scc_state;
  input.plan_token = kPlanToken;
  Gfn2EnergyForceExecutionDeviceResults results{};
  results.energy = device.results.energy;
  results.plan_token = kPlanToken;
  Gfn2EnergyForceExecutionDeviceIntermediates intermediates{};
  intermediates.energy = device.intermediates.energy;
  intermediates.plan_token = kPlanToken;
  Gfn2EnergyForceExecutionDeviceWorkspace workspace{};
  workspace.total_energy = device.workspace.total_energy;
  workspace.plan_failure = device.workspace.plan_failure;
  workspace.plan_failure_elements = 1;
  workspace.plan_token = kPlanToken;
  Gfn2EnergyForceExecutionDeviceDiagnostics diagnostics{};
  diagnostics.execution_system_errors = device.diagnostics.execution_system_errors;
  diagnostics.execution_device_error = device.diagnostics.execution_device_error;
  diagnostics.total_energy_system_errors = device.diagnostics.total_energy_system_errors;
  diagnostics.total_energy_device_error = device.diagnostics.total_energy_device_error;
  diagnostics.batch_elements = static_cast<std::int64_t>(host.batch_size);
  diagnostics.plan_token = kPlanToken;
  CUDA_CHECK(execute_gfn2_energy_force_cuda(plan, input, results, intermediates, workspace,
                                            diagnostics, stream));

  std::vector<double> energy(host.batch_size);
  std::vector<double> qm(host.expected_qm_force.size());
  CUDA_CHECK(device.public_energy.copy_to(energy.data(), energy.size(), stream));
  CUDA_CHECK(device.public_qm_force.copy_to(qm.data(), qm.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (std::size_t system = 0; system < energy.size(); ++system) {
    CHECK(near(energy[system], host.expected_energy[system]));
  }
  CHECK(std::all_of(qm.begin(), qm.end(), [](double value) { return value == kSentinel; }));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_peer_failure_transaction() {
  HostCase host;
  std::string error;
  CHECK(make_case(8u, host, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(initialize_device(device, host, stream));
  CUDA_CHECK(seed_public_results(device, host, stream));

  std::vector<std::uint8_t> requested(host.batch_size, 1u);
  requested[0] = 0u;
  std::vector<double> raw_shell = host.shell_charges;
  raw_shell[static_cast<std::size_t>(host.basis.batch_shell_offsets[0])] =
      std::numeric_limits<double>::quiet_NaN();
  raw_shell[static_cast<std::size_t>(host.basis.batch_shell_offsets[1])] =
      std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(device.requested.copy_from(requested.data(), requested.size(), stream));
  CUDA_CHECK(device.shell_charges.copy_from(raw_shell.data(), raw_shell.size(), stream));
  CUDA_CHECK(launch_execution(device, stream));

  std::vector<double> energy(host.batch_size);
  std::vector<double> qm(host.expected_qm_force.size());
  std::vector<double> point(host.expected_point_force.size());
  std::vector<std::uint32_t> execution_errors(host.batch_size);
  std::vector<std::uint32_t> post_errors(host.batch_size);
  std::uint32_t execution_device_error = 0u;
  std::uint32_t plan_failure = 1u;
  CUDA_CHECK(device.public_energy.copy_to(energy.data(), energy.size(), stream));
  CUDA_CHECK(device.public_qm_force.copy_to(qm.data(), qm.size(), stream));
  CUDA_CHECK(device.public_point_force.copy_to(point.data(), point.size(), stream));
  CUDA_CHECK(device.execution_system_errors.copy_to(execution_errors.data(),
                                                    execution_errors.size(), stream));
  CUDA_CHECK(device.post_system_errors.copy_to(post_errors.data(), post_errors.size(), stream));
  CUDA_CHECK(device.execution_device_error.copy_to(&execution_device_error, 1u, stream));
  CUDA_CHECK(device.plan_failure.copy_to(&plan_failure, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CHECK(plan_failure == 0u);
  CHECK(execution_device_error ==
        static_cast<std::uint32_t>(Gfn2EnergyForceExecutionDeviceError::kPostSccPotentialFailure));
  CHECK(execution_errors[0] == 0u);
  CHECK(execution_errors[1] ==
        static_cast<std::uint32_t>(Gfn2EnergyForceExecutionDeviceError::kPostSccPotentialFailure));
  CHECK(gfn2_post_scc_potential_error_stage(post_errors[1]) == Gfn2PostSccPotentialStage::kES2);
  for (std::size_t system = 0; system < host.batch_size; ++system) {
    const std::int64_t atom_begin = host.basis.atom_offsets[system];
    const std::int64_t atom_end = host.basis.atom_offsets[system + 1u];
    const std::int64_t point_begin = host.external_plan.point_charge_offsets[system];
    const std::int64_t point_end = host.external_plan.point_charge_offsets[system + 1u];
    if (system == 0u) {
      CHECK(near(energy[system], host.expected_energy[system]));
    } else if (system == 1u) {
      CHECK(energy[system] == kSentinel);
    } else {
      CHECK(near(energy[system], host.expected_energy[system]));
    }
    for (std::int64_t coordinate = 3 * atom_begin; coordinate < 3 * atom_end; ++coordinate) {
      const std::size_t index = static_cast<std::size_t>(coordinate);
      if (system <= 1u) {
        CHECK(qm[index] == kSentinel);
      } else {
        CHECK(near(qm[index], host.expected_qm_force[index], 2.0e-8, 2.0e-8));
      }
    }
    for (std::int64_t coordinate = 3 * point_begin; coordinate < 3 * point_end; ++coordinate) {
      const std::size_t index = static_cast<std::size_t>(coordinate);
      if (system <= 1u) {
        CHECK(point[index] == kSentinel);
      } else {
        CHECK(near(point[index], host.expected_point_force[index], 2.0e-9, 2.0e-9));
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_plan_failure_suppresses_publication() {
  HostCase host;
  std::string error;
  CHECK(make_case(8u, host, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(initialize_device(device, host, stream));
  CUDA_CHECK(seed_public_results(device, host, stream));

  std::vector<std::int64_t> invalid_qsh_offsets = host.basis.batch_shell_offsets;
  invalid_qsh_offsets[0] = 1;
  CUDA_CHECK(
      device.qsh_offsets.copy_from(invalid_qsh_offsets.data(), invalid_qsh_offsets.size(), stream));
  CUDA_CHECK(launch_execution(device, stream));

  std::vector<double> energy(host.batch_size);
  std::vector<double> qm(host.expected_qm_force.size());
  std::vector<double> point(host.expected_point_force.size());
  std::uint32_t execution_device_error = 0u;
  std::uint32_t plan_failure = 0u;
  std::uint32_t post_device_error = 0u;
  CUDA_CHECK(device.public_energy.copy_to(energy.data(), energy.size(), stream));
  CUDA_CHECK(device.public_qm_force.copy_to(qm.data(), qm.size(), stream));
  CUDA_CHECK(device.public_point_force.copy_to(point.data(), point.size(), stream));
  CUDA_CHECK(device.execution_device_error.copy_to(&execution_device_error, 1u, stream));
  CUDA_CHECK(device.plan_failure.copy_to(&plan_failure, 1u, stream));
  CUDA_CHECK(device.post_device_error.copy_to(&post_device_error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CHECK(execution_device_error ==
        static_cast<std::uint32_t>(Gfn2EnergyForceExecutionDeviceError::kPostSccPotentialFailure));
  CHECK(plan_failure ==
        static_cast<std::uint32_t>(Gfn2EnergyForceExecutionDeviceError::kPostSccPotentialFailure));
  CHECK(gfn2_post_scc_potential_error_stage(post_device_error) ==
        Gfn2PostSccPotentialStage::kComposition);
  CHECK(std::all_of(energy.begin(), energy.end(), [](double value) { return value == kSentinel; }));
  CHECK(std::all_of(qm.begin(), qm.end(), [](double value) { return value == kSentinel; }));
  CHECK(std::all_of(point.begin(), point.end(), [](double value) { return value == kSentinel; }));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_graph_replay_uses_changed_raw_state() {
  HostCase initial;
  std::string error;
  CHECK(make_case(8u, initial, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(initialize_device(device, initial, stream));
  CUDA_CHECK(seed_public_results(device, initial, stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CUDA_CHECK(launch_execution(device, stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  if (const int line = compare_success(device, initial, stream); line != 0) {
    return line;
  }

  HostCase changed;
  CHECK(make_case(8u, changed, error));
  for (std::size_t shell = 0; shell < changed.shell_charges.size(); ++shell) {
    changed.shell_charges[shell] =
        -0.55 * changed.shell_charges[shell] + 0.002 * static_cast<double>(shell + 1u);
  }
  std::fill(changed.atomic_charges.begin(), changed.atomic_charges.end(), 0.0);
  for (std::size_t shell = 0; shell < changed.shell_charges.size(); ++shell) {
    const std::size_t atom =
        static_cast<std::size_t>(changed.basis.shell_to_atom[static_cast<std::size_t>(shell)]);
    changed.atomic_charges[atom] += changed.shell_charges[shell];
  }
  for (std::size_t index = 0; index < changed.atomic_dipoles.size(); ++index) {
    changed.atomic_dipoles[index] += 0.001 * static_cast<double>(index % 5u + 1u);
  }
  for (std::size_t index = 0; index < changed.atomic_quadrupoles.size(); ++index) {
    changed.atomic_quadrupoles[index] -= 0.0003 * static_cast<double>(index % 7u + 1u);
  }
  CHECK(refresh_physics(changed, error));
  CUDA_CHECK(seed_public_results(device, changed, stream));
  CUDA_CHECK(upload_raw_state(device, changed, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  if (const int line = compare_success(device, changed, stream); line != 0) {
    return line;
  }

  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main() {
  for (std::size_t batch : {1u, 8u, 32u, 128u}) {
    if (const int line = test_batch_parity(batch); line != 0) {
      return line;
    }
  }
  if (const int line = test_energy_only_ignores_force_bindings(); line != 0) {
    return line;
  }
  if (const int line = test_peer_failure_transaction(); line != 0) {
    return line;
  }
  if (const int line = test_plan_failure_suppresses_publication(); line != 0) {
    return line;
  }
  return test_graph_replay_uses_changed_raw_state();
}
