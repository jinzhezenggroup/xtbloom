#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_classical_force.cuh"
#include "backends/cuda/gfn2_parameters.cuh"
#include "data/parameters/d4.hpp"
#include "model/gfn2/aes2.hpp"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/coordination.hpp"
#include "model/gfn2/d4.hpp"
#include "model/gfn2/es2.hpp"
#include "model/gfn2/repulsion.hpp"
#include "runtime/backend.hpp"

#define CHECK(condition)                                                                           \
  do {                                                                                             \
    if (!(condition)) {                                                                            \
      std::fprintf(stderr, "classical force check failed at line %d: %s\n", __LINE__, #condition); \
      return __LINE__;                                                                             \
    }                                                                                              \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using namespace xtbloom::detail;
using namespace xtbloom::detail::cuda;

constexpr std::uint64_t kPlanToken = 0x651165116511ULL;
constexpr std::uint64_t kGeneration = 65u;

bool near(double actual, double expected, double tolerance = 2.0e-10) {
  return std::abs(actual - expected) <=
         tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

bool component_enabled(std::uint32_t mask, Gfn2ClassicalForceComponent component) {
  return (mask & static_cast<std::uint32_t>(component)) != 0u;
}

using Rotation = std::array<double, 9>;

Rotation make_test_rotation() {
  constexpr double angle = 0.73;
  constexpr double inverse_norm = 0.40824829046386301637;
  const std::array<double, 3> axis{inverse_norm, 2.0 * inverse_norm, -inverse_norm};
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  const double complement = 1.0 - cosine;
  return {
      cosine + axis[0] * axis[0] * complement,
      axis[0] * axis[1] * complement - axis[2] * sine,
      axis[0] * axis[2] * complement + axis[1] * sine,
      axis[1] * axis[0] * complement + axis[2] * sine,
      cosine + axis[1] * axis[1] * complement,
      axis[1] * axis[2] * complement - axis[0] * sine,
      axis[2] * axis[0] * complement - axis[1] * sine,
      axis[2] * axis[1] * complement + axis[0] * sine,
      cosine + axis[2] * axis[2] * complement,
  };
}

std::array<double, 3> rotate_vector(const Rotation& rotation, const double* vector) {
  std::array<double, 3> rotated{};
  for (std::size_t row = 0; row < 3u; ++row) {
    for (std::size_t column = 0; column < 3u; ++column) {
      rotated[row] += rotation[row * 3u + column] * vector[column];
    }
  }
  return rotated;
}

std::array<double, 6> rotate_quadrupole(const Rotation& rotation, const double* packed) {
  const double tensor[3][3] = {
      {packed[0], packed[1], packed[3]},
      {packed[1], packed[2], packed[4]},
      {packed[3], packed[4], packed[5]},
  };
  double rotated[3][3]{};
  for (std::size_t row = 0; row < 3u; ++row) {
    for (std::size_t column = 0; column < 3u; ++column) {
      for (std::size_t first = 0; first < 3u; ++first) {
        for (std::size_t second = 0; second < 3u; ++second) {
          rotated[row][column] +=
              rotation[row * 3u + first] * tensor[first][second] * rotation[column * 3u + second];
        }
      }
    }
  }
  return {rotated[0][0], rotated[0][1], rotated[1][1], rotated[0][2], rotated[1][2], rotated[2][2]};
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
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
    count_ = count;
    return count == 0u ||
           cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)) == cudaSuccess;
  }

  bool copy_from(const T* source, std::size_t count, cudaStream_t stream) {
    return count <= count_ && (count == 0u || source != nullptr) &&
           (count == 0u || cudaMemcpyAsync(data_, source, count * sizeof(T), cudaMemcpyHostToDevice,
                                           stream) == cudaSuccess);
  }

  bool copy_to(T* target, std::size_t count, cudaStream_t stream) const {
    return count <= count_ && (count == 0u || target != nullptr) &&
           (count == 0u || cudaMemcpyAsync(target, data_, count * sizeof(T), cudaMemcpyDeviceToHost,
                                           stream) == cudaSuccess);
  }

  bool copy_device_from(const DeviceBuffer& source, std::size_t count, cudaStream_t stream) {
    return count <= count_ && count <= source.count_ &&
           (count == 0u || cudaMemcpyAsync(data_, source.data_, count * sizeof(T),
                                           cudaMemcpyDeviceToDevice, stream) == cudaSuccess);
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

template <typename T>
bool allocate_and_copy(DeviceBuffer<T>& device, const std::vector<T>& host, cudaStream_t stream) {
  return device.allocate(host.size()) && device.copy_from(host.data(), host.size(), stream);
}

struct HostFixture {
  std::size_t batch_size = 0u;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> coordination;
  std::vector<double> shell_charges;
  std::vector<double> atomic_charges;
  std::vector<double> dipoles;
  std::vector<double> quadrupoles;
  std::vector<double> force_seed;

  gfn2::BasisPlan basis;
  gfn2::CoordinationPlan coordination_plan;
  gfn2::RepulsionPlan repulsion_plan;
  gfn2::ES2Plan es2_plan;
  gfn2::AES2Plan aes2_plan;
  gfn2::D4Plan d4_plan;

  std::vector<double> es2_matrix;
  std::vector<double> es2_matrix_scratch;
  std::vector<double> es2_shell_scratch;
  std::vector<double> es2_batch_scratch;
  std::vector<double> es2_gradient_scratch;
  gfn2::ES2Workspace es2_workspace{};
  gfn2::ES2GeometryCache es2_cache{};

  std::vector<double> aes2_pairs;
  std::vector<double> aes2_pair_scratch;
  std::vector<double> aes2_potential_scratch;
  std::vector<double> aes2_batch_scratch;
  std::vector<double> aes2_gradient_scratch;
  std::vector<double> aes2_coordination_scratch;
  gfn2::AES2Workspace aes2_workspace{};
  gfn2::AES2GeometryCache aes2_cache{};

  std::vector<std::byte> d4_workspace_storage;
  gfn2::D4Workspace d4_workspace{};
  std::vector<double> d4_pairs;
  std::vector<double> d4_coordination;
  gfn2::D4GeometryCache d4_cache{};

  bool refresh_geometry(std::string& error) {
    return gfn2::evaluate_coordination_cpu(coordination_plan, positions.data(), coordination.data(),
                                           error) == XTBLOOM_STATUS_SUCCESS &&
           gfn2::update_es2_geometry_cache_cpu(es2_plan, positions.data(), kGeneration,
                                               es2_matrix.data(), es2_matrix.size(), es2_workspace,
                                               es2_cache, error) == XTBLOOM_STATUS_SUCCESS &&
           gfn2::update_aes2_geometry_cache_cpu(
               aes2_plan, positions.data(), coordination.data(), kGeneration, aes2_pairs.data(),
               aes2_pairs.size(), aes2_workspace, aes2_cache, error) == XTBLOOM_STATUS_SUCCESS &&
           gfn2::update_d4_geometry_cache_cpu(
               d4_plan, positions.data(), kGeneration, d4_pairs.data(), d4_pairs.size(),
               d4_coordination.data(), d4_coordination.size(), d4_workspace, d4_cache,
               error) == XTBLOOM_STATUS_SUCCESS;
  }

  bool initialize(std::size_t requested_batch, std::string& error) {
    batch_size = requested_batch;
    atom_offsets.resize(batch_size + 1u);
    atom_offsets[0] = 0;
    for (std::size_t system = 0; system < batch_size; ++system) {
      const double shift = 8.0 * static_cast<double>(system);
      atomic_numbers.insert(atomic_numbers.end(), {6, 1, 8});
      positions.insert(positions.end(),
                       {shift, 0.05, -0.1, shift + 1.53, 0.87, 0.31, shift - 1.17, 1.26, -0.42});
      atom_offsets[system + 1u] = static_cast<std::int64_t>(atomic_numbers.size());
    }
    const std::size_t atoms = atomic_numbers.size();
    coordination.resize(atoms);
    atomic_charges.resize(atoms);
    dipoles.resize(atoms * 3u);
    quadrupoles.resize(atoms * 6u);
    force_seed.resize(atoms * 3u);
    for (std::size_t atom = 0; atom < atoms; ++atom) {
      atomic_charges[atom] = 0.035 * static_cast<double>(static_cast<int>(atom % 5u) - 2);
      for (int axis = 0; axis < 3; ++axis) {
        dipoles[atom * 3u + static_cast<std::size_t>(axis)] =
            0.004 * static_cast<double>(static_cast<int>((atom * 3u + axis) % 7u) - 3);
        force_seed[atom * 3u + static_cast<std::size_t>(axis)] =
            0.0007 * static_cast<double>(static_cast<int>((atom * 3u + axis) % 9u) - 4);
      }
      for (int component = 0; component < 6; ++component) {
        quadrupoles[atom * 6u + static_cast<std::size_t>(component)] =
            0.0015 * static_cast<double>(static_cast<int>((atom * 6u + component) % 11u) - 5);
      }
    }

    const std::int64_t batch = static_cast<std::int64_t>(batch_size);
    const std::int64_t total_atoms = static_cast<std::int64_t>(atoms);
    if (gfn2::make_basis_plan(batch, total_atoms, atom_offsets.data(), atomic_numbers.data(), basis,
                              error) != XTBLOOM_STATUS_SUCCESS ||
        gfn2::make_coordination_plan(batch, total_atoms, atom_offsets.data(), atomic_numbers.data(),
                                     coordination_plan, error) != XTBLOOM_STATUS_SUCCESS ||
        gfn2::make_repulsion_plan(batch, total_atoms, atom_offsets.data(), atomic_numbers.data(),
                                  repulsion_plan, error) != XTBLOOM_STATUS_SUCCESS ||
        gfn2::make_es2_plan(basis, atomic_numbers.data(), es2_plan, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        gfn2::make_aes2_plan(basis, atomic_numbers.data(), aes2_plan, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        gfn2::make_d4_plan(batch, total_atoms, atom_offsets.data(), atomic_numbers.data(), d4_plan,
                           error) != XTBLOOM_STATUS_SUCCESS ||
        gfn2::evaluate_coordination_cpu(coordination_plan, positions.data(), coordination.data(),
                                        error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }

    shell_charges.resize(static_cast<std::size_t>(es2_plan.total_shells()));
    for (std::size_t shell = 0; shell < shell_charges.size(); ++shell) {
      shell_charges[shell] = 0.018 * static_cast<double>(static_cast<int>(shell % 7u) - 3);
    }

    es2_matrix.resize(static_cast<std::size_t>(es2_plan.total_matrix_elements()));
    es2_matrix_scratch.resize(es2_matrix.size());
    es2_shell_scratch.resize(shell_charges.size());
    es2_batch_scratch.resize(batch_size);
    es2_gradient_scratch.resize(atoms * 3u);
    es2_workspace = {es2_matrix_scratch.data(),   es2_plan.total_matrix_elements(),
                     es2_shell_scratch.data(),    es2_plan.total_shells(),
                     es2_batch_scratch.data(),    static_cast<std::int64_t>(batch_size),
                     es2_gradient_scratch.data(), static_cast<std::int64_t>(atoms * 3u)};
    if (gfn2::update_es2_geometry_cache_cpu(es2_plan, positions.data(), kGeneration,
                                            es2_matrix.data(), es2_matrix.size(), es2_workspace,
                                            es2_cache, error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }

    aes2_pairs.resize(static_cast<std::size_t>(aes2_plan.pair_data_elements()));
    aes2_pair_scratch.resize(aes2_pairs.size());
    aes2_potential_scratch.resize(static_cast<std::size_t>(aes2_plan.potential_scratch_elements()));
    aes2_batch_scratch.resize(batch_size);
    aes2_gradient_scratch.resize(atoms * 3u);
    aes2_coordination_scratch.resize(atoms);
    aes2_workspace = {aes2_pair_scratch.data(),         aes2_plan.pair_data_elements(),
                      aes2_potential_scratch.data(),    aes2_plan.potential_scratch_elements(),
                      aes2_batch_scratch.data(),        static_cast<std::int64_t>(batch_size),
                      aes2_gradient_scratch.data(),     static_cast<std::int64_t>(atoms * 3u),
                      aes2_coordination_scratch.data(), static_cast<std::int64_t>(atoms)};
    if (gfn2::update_aes2_geometry_cache_cpu(
            aes2_plan, positions.data(), coordination.data(), kGeneration, aes2_pairs.data(),
            aes2_pairs.size(), aes2_workspace, aes2_cache, error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }

    d4_workspace_storage.resize(d4_plan.workspace_size_bytes() + gfn2::kD4WorkspaceAlignment);
    const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(d4_workspace_storage.data());
    const std::uintptr_t aligned =
        (raw + gfn2::kD4WorkspaceAlignment - 1u) & ~(gfn2::kD4WorkspaceAlignment - 1u);
    if (gfn2::bind_d4_workspace(d4_plan, reinterpret_cast<void*>(aligned),
                                d4_plan.workspace_size_bytes(), d4_workspace,
                                error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    d4_pairs.resize(static_cast<std::size_t>(d4_plan.total_pairs()) * gfn2::kD4PairDataElements);
    d4_coordination.resize(atoms);
    return gfn2::update_d4_geometry_cache_cpu(
               d4_plan, positions.data(), kGeneration, d4_pairs.data(), d4_pairs.size(),
               d4_coordination.data(), d4_coordination.size(), d4_workspace, d4_cache,
               error) == XTBLOOM_STATUS_SUCCESS;
  }

  bool expected_forces(std::uint32_t mask, std::vector<double>& forces, std::string& error) {
    forces = force_seed;
    std::vector<double> gradient(forces.size(), 0.0);
    if (component_enabled(mask, Gfn2ClassicalForceComponent::kRepulsion)) {
      std::vector<double> energies(batch_size, 0.0);
      std::vector<double> contribution(forces.size(), 0.0);
      if (gfn2::add_repulsion_cpu(repulsion_plan, positions.data(), energies.data(),
                                  contribution.data(), error) != XTBLOOM_STATUS_SUCCESS) {
        return false;
      }
      for (std::size_t index = 0; index < forces.size(); ++index) {
        forces[index] += contribution[index];
      }
    }
    if (component_enabled(mask, Gfn2ClassicalForceComponent::kES2)) {
      std::fill(gradient.begin(), gradient.end(), 0.0);
      if (gfn2::add_es2_gradient_cpu(es2_plan, es2_cache, positions.data(), kGeneration,
                                     shell_charges.data(), gradient.data(), es2_workspace,
                                     error) != XTBLOOM_STATUS_SUCCESS) {
        return false;
      }
      for (std::size_t index = 0; index < forces.size(); ++index) {
        forces[index] -= gradient[index];
      }
    }
    if (component_enabled(mask, Gfn2ClassicalForceComponent::kAES2)) {
      std::fill(gradient.begin(), gradient.end(), 0.0);
      std::vector<double> cn(atomic_numbers.size(), 0.0);
      if (gfn2::add_aes2_vjp_cpu(aes2_plan, aes2_cache, positions.data(), coordination.data(),
                                 kGeneration, atomic_charges.data(), dipoles.data(),
                                 quadrupoles.data(), gradient.data(), cn.data(), aes2_workspace,
                                 error) != XTBLOOM_STATUS_SUCCESS ||
          gfn2::add_coordination_gradient_cpu(coordination_plan, positions.data(), cn.data(),
                                              gradient.data(), error) != XTBLOOM_STATUS_SUCCESS) {
        return false;
      }
      for (std::size_t index = 0; index < forces.size(); ++index) {
        forces[index] -= gradient[index];
      }
    }
    if (component_enabled(mask, Gfn2ClassicalForceComponent::kD4TwoBody)) {
      std::fill(gradient.begin(), gradient.end(), 0.0);
      if (gfn2::add_d4_two_body_gradient_cpu(d4_plan, d4_cache, atomic_charges.data(),
                                             gradient.data(), d4_workspace,
                                             error) != XTBLOOM_STATUS_SUCCESS) {
        return false;
      }
      for (std::size_t index = 0; index < forces.size(); ++index) {
        forces[index] -= gradient[index];
      }
    }
    if (component_enabled(mask, Gfn2ClassicalForceComponent::kD4ATM)) {
      std::fill(gradient.begin(), gradient.end(), 0.0);
      if (gfn2::add_d4_atm_gradient_cpu(d4_plan, d4_cache, gradient.data(), d4_workspace, error) !=
          XTBLOOM_STATUS_SUCCESS) {
        return false;
      }
      for (std::size_t index = 0; index < forces.size(); ++index) {
        forces[index] -= gradient[index];
      }
    }
    return true;
  }

  /* Complete fixed-multipole classical energy used for a central-difference gate. */
  bool total_energy_at(const std::vector<double>& displaced_positions, double& energy,
                       std::string& error) {
    if (batch_size != 1u || displaced_positions.size() != positions.size()) {
      return false;
    }
    std::vector<double> displaced_coordination(coordination.size());
    if (gfn2::evaluate_coordination_cpu(coordination_plan, displaced_positions.data(),
                                        displaced_coordination.data(),
                                        error) != XTBLOOM_STATUS_SUCCESS ||
        gfn2::update_es2_geometry_cache_cpu(es2_plan, displaced_positions.data(), kGeneration,
                                            es2_matrix.data(), es2_matrix.size(), es2_workspace,
                                            es2_cache, error) != XTBLOOM_STATUS_SUCCESS ||
        gfn2::update_aes2_geometry_cache_cpu(aes2_plan, displaced_positions.data(),
                                             displaced_coordination.data(), kGeneration,
                                             aes2_pairs.data(), aes2_pairs.size(), aes2_workspace,
                                             aes2_cache, error) != XTBLOOM_STATUS_SUCCESS ||
        gfn2::update_d4_geometry_cache_cpu(d4_plan, displaced_positions.data(), kGeneration,
                                           d4_pairs.data(), d4_pairs.size(), d4_coordination.data(),
                                           d4_coordination.size(), d4_workspace, d4_cache,
                                           error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    double repulsion = 0.0;
    double es2 = 0.0;
    double aes2 = 0.0;
    double d4_two_body = 0.0;
    double d4_atm = 0.0;
    std::vector<double> d4_potential(atomic_numbers.size());
    if (gfn2::add_repulsion_cpu(repulsion_plan, displaced_positions.data(), &repulsion, nullptr,
                                error) != XTBLOOM_STATUS_SUCCESS ||
        gfn2::add_es2_energy_cpu(es2_plan, es2_cache, shell_charges.data(), &es2, es2_workspace,
                                 error) != XTBLOOM_STATUS_SUCCESS ||
        gfn2::add_aes2_energy_cpu(aes2_plan, aes2_cache, atomic_charges.data(), dipoles.data(),
                                  quadrupoles.data(), &aes2, aes2_workspace,
                                  error) != XTBLOOM_STATUS_SUCCESS ||
        gfn2::evaluate_d4_two_body_cpu(d4_plan, d4_cache, atomic_charges.data(), &d4_two_body,
                                       d4_potential.data(), d4_workspace,
                                       error) != XTBLOOM_STATUS_SUCCESS ||
        gfn2::evaluate_d4_atm_cpu(d4_plan, d4_cache, &d4_atm, d4_workspace, error) !=
            XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    energy = repulsion + es2 + aes2 + d4_two_body + d4_atm;
    return std::isfinite(energy);
  }
};

struct DeviceFixture {
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> pair_offsets;
  DeviceBuffer<std::int64_t> batch_shell_offsets;
  DeviceBuffer<std::int64_t> atom_shell_offsets;
  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<std::int64_t> shell_to_atom;
  DeviceBuffer<std::int32_t> atomic_numbers;
  DeviceBuffer<double> positions;
  DeviceBuffer<double> coordination;
  DeviceBuffer<double> shell_charges;
  DeviceBuffer<double> atomic_charges;
  DeviceBuffer<double> dipoles;
  DeviceBuffer<double> quadrupoles;
  DeviceBuffer<double> force_seed;
  DeviceBuffer<double> forces;

  DeviceBuffer<double> covalent_radii;
  DeviceBuffer<double> geometry_pairs;
  DeviceBuffer<double> geometry_cn;
  DeviceBuffer<std::uint64_t> geometry_generations;
  DeviceBuffer<double> geometry_pair_scratch;
  DeviceBuffer<double> geometry_cn_scratch;
  DeviceBuffer<double> geometry_gradient_scratch;
  DeviceBuffer<std::uint32_t> geometry_sequence;

  DeviceBuffer<double> es2_hardness;
  DeviceBuffer<double> es2_matrix;
  DeviceBuffer<double> es2_matrix_scratch;

  DeviceBuffer<double> aes2_dipole_kernel;
  DeviceBuffer<double> aes2_quadrupole_kernel;
  DeviceBuffer<double> aes2_radius;
  DeviceBuffer<double> aes2_valence_cn;
  DeviceBuffer<double> aes2_pairs;
  DeviceBuffer<double> aes2_pair_scratch;
  DeviceBuffer<double> aes2_gradient_scratch;
  DeviceBuffer<double> aes2_cn_scratch;

  DeviceBuffer<Gfn2D4DeviceElementData> d4_elements;
  DeviceBuffer<Gfn2D4DeviceReferenceData> d4_references;
  DeviceBuffer<double> d4_reference_c6;
  DeviceBuffer<double> d4_pairs;
  DeviceBuffer<double> d4_coordination;
  DeviceBuffer<double> d4_weights;
  DeviceBuffer<double> d4_weight_cn;
  DeviceBuffer<double> d4_weight_charge;
  DeviceBuffer<double> d4_atom_scratch;
  DeviceBuffer<double> d4_cn_adjoints;
  DeviceBuffer<double> d4_batch_scratch;
  DeviceBuffer<double> d4_gradient_scratch;

  DeviceBuffer<double> gradient_scratch;
  DeviceBuffer<double> force_scratch;
  DeviceBuffer<double> coordination_adjoints;
  DeviceBuffer<std::uint8_t> requested;
  DeviceBuffer<xtbloom_status_t> statuses;
  DeviceBuffer<std::uint8_t> selected;
  DeviceBuffer<std::uint32_t> primitive_system_errors;
  DeviceBuffer<std::uint32_t> primitive_device_error;
  DeviceBuffer<std::uint32_t> sequence_active;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;

  Gfn2ClassicalForceDevicePlan plan{};
  Gfn2ClassicalForceDeviceInput input{};
  Gfn2ClassicalForceDeviceOutput output{};
  Gfn2ClassicalForceDeviceWorkspace workspace{};
  Gfn2ForceDeviceActivity activity{};

  bool initialize(const HostFixture& host, cudaStream_t stream) {
    const std::size_t batch = host.batch_size;
    const std::size_t atoms = host.atomic_numbers.size();
    const std::size_t shells = host.shell_charges.size();
    const std::size_t pairs = static_cast<std::size_t>(host.aes2_plan.total_pairs());
    const std::size_t coordinates = atoms * 3u;
    std::vector<Gfn2D4DeviceElementData> elements;
    for (const auto& value : xtbloom::parameters::d4::kElements) {
      elements.push_back({value.reference_offset, value.reference_count, value.covalent_radius,
                          value.electronegativity, value.effective_charge, value.hardness,
                          value.r4r2});
    }
    std::vector<Gfn2D4DeviceReferenceData> references;
    for (const auto& value : xtbloom::parameters::d4::kReferences) {
      references.push_back({value.coordination_number, value.charge, value.gaussian_count});
    }
    std::vector<std::uint64_t> generations(batch, kGeneration);
    std::vector<std::uint8_t> requested_host(batch, 1u);
    std::vector<xtbloom_status_t> statuses_host(batch, XTBLOOM_STATUS_SUCCESS);

    if (!allocate_and_copy(atom_offsets, host.atom_offsets, stream) ||
        !allocate_and_copy(pair_offsets, host.aes2_plan.pair_offsets(), stream) ||
        !allocate_and_copy(batch_shell_offsets, host.es2_plan.batch_shell_offsets(), stream) ||
        !allocate_and_copy(atom_shell_offsets, host.es2_plan.atom_shell_offsets(), stream) ||
        !allocate_and_copy(matrix_offsets, host.es2_plan.matrix_offsets(), stream) ||
        !allocate_and_copy(shell_to_atom, host.es2_plan.shell_to_atom(), stream) ||
        !allocate_and_copy(atomic_numbers, host.atomic_numbers, stream) ||
        !allocate_and_copy(positions, host.positions, stream) ||
        !allocate_and_copy(coordination, host.coordination, stream) ||
        !allocate_and_copy(shell_charges, host.shell_charges, stream) ||
        !allocate_and_copy(atomic_charges, host.atomic_charges, stream) ||
        !allocate_and_copy(dipoles, host.dipoles, stream) ||
        !allocate_and_copy(quadrupoles, host.quadrupoles, stream) ||
        !allocate_and_copy(force_seed, host.force_seed, stream) ||
        !allocate_and_copy(covalent_radii, host.coordination_plan.covalent_radius, stream) ||
        !allocate_and_copy(geometry_generations, generations, stream) ||
        !allocate_and_copy(es2_hardness, host.es2_plan.shell_hardness(), stream) ||
        !allocate_and_copy(aes2_dipole_kernel, host.aes2_plan.dipole_kernel(), stream) ||
        !allocate_and_copy(aes2_quadrupole_kernel, host.aes2_plan.quadrupole_kernel(), stream) ||
        !allocate_and_copy(aes2_radius, host.aes2_plan.multipole_radius(), stream) ||
        !allocate_and_copy(aes2_valence_cn, host.aes2_plan.multipole_valence_cn(), stream) ||
        !allocate_and_copy(d4_elements, elements, stream) ||
        !allocate_and_copy(d4_references, references, stream) ||
        !d4_reference_c6.allocate(xtbloom::parameters::d4::kReferenceC6.size()) ||
        !d4_reference_c6.copy_from(xtbloom::parameters::d4::kReferenceC6.data(),
                                   xtbloom::parameters::d4::kReferenceC6.size(), stream) ||
        !allocate_and_copy(d4_pairs, host.d4_pairs, stream) ||
        !allocate_and_copy(d4_coordination, host.d4_coordination, stream) ||
        !allocate_and_copy(requested, requested_host, stream) ||
        !allocate_and_copy(statuses, statuses_host, stream) || !forces.allocate(coordinates) ||
        !forces.copy_from(host.force_seed.data(), coordinates, stream) ||
        !geometry_pairs.allocate(pairs * kGfn2GeometryPairDataElements) ||
        !geometry_cn.allocate(atoms) ||
        !geometry_pair_scratch.allocate(pairs * kGfn2GeometryPairDataElements) ||
        !geometry_cn_scratch.allocate(atoms) || !geometry_gradient_scratch.allocate(coordinates) ||
        !geometry_sequence.allocate(1) ||
        !es2_matrix.allocate(static_cast<std::size_t>(host.es2_plan.total_matrix_elements())) ||
        !es2_matrix_scratch.allocate(
            static_cast<std::size_t>(host.es2_plan.total_matrix_elements())) ||
        !aes2_pairs.allocate(pairs * kGfn2AES2PairDataElements) ||
        !aes2_pair_scratch.allocate(pairs * kGfn2AES2PairDataElements) ||
        !aes2_gradient_scratch.allocate(coordinates) || !aes2_cn_scratch.allocate(atoms) ||
        !d4_weights.allocate(atoms * static_cast<std::size_t>(kGfn2D4MaximumReferences)) ||
        !d4_weight_cn.allocate(atoms * static_cast<std::size_t>(kGfn2D4MaximumReferences)) ||
        !d4_weight_charge.allocate(atoms * static_cast<std::size_t>(kGfn2D4MaximumReferences)) ||
        !d4_atom_scratch.allocate(atoms) || !d4_cn_adjoints.allocate(atoms) ||
        !d4_batch_scratch.allocate(batch) || !d4_gradient_scratch.allocate(coordinates) ||
        !gradient_scratch.allocate(coordinates) || !force_scratch.allocate(coordinates) ||
        !coordination_adjoints.allocate(atoms) || !selected.allocate(batch) ||
        !primitive_system_errors.allocate(batch) || !primitive_device_error.allocate(1) ||
        !sequence_active.allocate(1) || !system_errors.allocate(batch) ||
        !device_error.allocate(1)) {
      return false;
    }

    Gfn2GeometryDeviceBatch geometry_batch{static_cast<std::int64_t>(batch),
                                           static_cast<std::int64_t>(atoms),
                                           static_cast<std::int64_t>(pairs),
                                           static_cast<std::int64_t>(batch + 1u),
                                           static_cast<std::int64_t>(batch + 1u),
                                           static_cast<std::int64_t>(atoms),
                                           static_cast<std::int64_t>(coordinates),
                                           kPlanToken,
                                           atom_offsets.get(),
                                           pair_offsets.get(),
                                           covalent_radii.get()};
    Gfn2GeometryDeviceCache geometry_cache{geometry_pairs.get(),
                                           static_cast<std::int64_t>(geometry_pairs.size()),
                                           geometry_cn.get(),
                                           static_cast<std::int64_t>(atoms),
                                           geometry_generations.get(),
                                           static_cast<std::int64_t>(batch),
                                           kPlanToken};
    Gfn2GeometryDeviceWorkspace geometry_workspace{
        geometry_pair_scratch.get(),
        static_cast<std::int64_t>(geometry_pair_scratch.size()),
        geometry_cn_scratch.get(),
        static_cast<std::int64_t>(atoms),
        geometry_gradient_scratch.get(),
        static_cast<std::int64_t>(coordinates),
        geometry_sequence.get(),
        1,
        kPlanToken};

    Gfn2ES2DeviceBatch es2_batch{static_cast<std::int64_t>(batch),
                                 static_cast<std::int64_t>(atoms),
                                 static_cast<std::int64_t>(shells),
                                 host.es2_plan.total_matrix_elements(),
                                 kPlanToken,
                                 static_cast<std::int64_t>(batch + 1u),
                                 static_cast<std::int64_t>(batch + 1u),
                                 static_cast<std::int64_t>(atoms + 1u),
                                 static_cast<std::int64_t>(batch + 1u),
                                 static_cast<std::int64_t>(shells),
                                 static_cast<std::int64_t>(shells),
                                 atom_offsets.get(),
                                 batch_shell_offsets.get(),
                                 atom_shell_offsets.get(),
                                 matrix_offsets.get(),
                                 shell_to_atom.get(),
                                 es2_hardness.get()};
    Gfn2ES2DeviceCache es2_cache{es2_matrix.get(), host.es2_plan.total_matrix_elements(),
                                 kGeneration, kPlanToken};
    Gfn2ES2DeviceWorkspace es2_workspace{es2_matrix_scratch.get(),
                                         host.es2_plan.total_matrix_elements(),
                                         nullptr,
                                         0,
                                         nullptr,
                                         0,
                                         nullptr,
                                         0};

    Gfn2AES2DeviceBatch aes2_batch{static_cast<std::int64_t>(batch),
                                   static_cast<std::int64_t>(atoms),
                                   static_cast<std::int64_t>(pairs),
                                   kPlanToken,
                                   static_cast<std::int64_t>(batch + 1u),
                                   static_cast<std::int64_t>(batch + 1u),
                                   static_cast<std::int64_t>(atoms),
                                   static_cast<std::int64_t>(atoms),
                                   static_cast<std::int64_t>(atoms),
                                   static_cast<std::int64_t>(atoms),
                                   atom_offsets.get(),
                                   pair_offsets.get(),
                                   aes2_dipole_kernel.get(),
                                   aes2_quadrupole_kernel.get(),
                                   aes2_radius.get(),
                                   aes2_valence_cn.get()};
    Gfn2AES2DeviceCache aes2_cache{aes2_pairs.get(), static_cast<std::int64_t>(aes2_pairs.size()),
                                   kGeneration, kPlanToken};
    Gfn2AES2DeviceWorkspace aes2_workspace{aes2_pair_scratch.get(),
                                           static_cast<std::int64_t>(aes2_pair_scratch.size()),
                                           nullptr,
                                           0,
                                           nullptr,
                                           0,
                                           aes2_gradient_scratch.get(),
                                           static_cast<std::int64_t>(coordinates),
                                           aes2_cn_scratch.get(),
                                           static_cast<std::int64_t>(atoms),
                                           nullptr,
                                           0};

    Gfn2D4DeviceBatch d4_batch{
        static_cast<std::int64_t>(batch),
        static_cast<std::int64_t>(atoms),
        static_cast<std::int64_t>(pairs),
        kPlanToken,
        gfn2_d4_atomic_number_hash(host.atomic_numbers.data(), static_cast<std::int64_t>(atoms)),
        atom_offsets.get(),
        pair_offsets.get(),
        atomic_numbers.get()};
    Gfn2D4DeviceParameters d4_parameters{
        d4_elements.get(),
        static_cast<std::int64_t>(elements.size()),
        d4_references.get(),
        static_cast<std::int64_t>(references.size()),
        d4_reference_c6.get(),
        static_cast<std::int64_t>(xtbloom::parameters::d4::kReferenceC6.size())};
    Gfn2D4DeviceCache d4_cache{d4_pairs.get(),        static_cast<std::int64_t>(d4_pairs.size()),
                               d4_coordination.get(), static_cast<std::int64_t>(atoms),
                               kGeneration,           kPlanToken};
    Gfn2D4DeviceWorkspace d4_workspace{d4_weights.get(),
                                       d4_weight_cn.get(),
                                       d4_weight_charge.get(),
                                       static_cast<std::int64_t>(d4_weights.size()),
                                       d4_atom_scratch.get(),
                                       d4_cn_adjoints.get(),
                                       static_cast<std::int64_t>(atoms),
                                       d4_batch_scratch.get(),
                                       static_cast<std::int64_t>(batch),
                                       d4_gradient_scratch.get(),
                                       static_cast<std::int64_t>(coordinates),
                                       primitive_system_errors.get(),
                                       static_cast<std::int64_t>(batch)};

    plan = {static_cast<std::int64_t>(batch),
            static_cast<std::int64_t>(atoms),
            static_cast<std::int64_t>(shells),
            kGfn2ClassicalForceAllComponents,
            kGeneration,
            kPlanToken,
            atom_offsets.get(),
            atomic_numbers.get(),
            geometry_batch,
            geometry_cache,
            es2_batch,
            es2_cache,
            aes2_batch,
            aes2_cache,
            d4_batch,
            d4_parameters,
            d4_cache};
    input = {positions.get(),
             static_cast<std::int64_t>(coordinates),
             coordination.get(),
             static_cast<std::int64_t>(atoms),
             shell_charges.get(),
             static_cast<std::int64_t>(shells),
             atomic_charges.get(),
             static_cast<std::int64_t>(atoms),
             dipoles.get(),
             static_cast<std::int64_t>(coordinates),
             quadrupoles.get(),
             static_cast<std::int64_t>(atoms * 6u),
             kPlanToken};
    output = {forces.get(), static_cast<std::int64_t>(coordinates), kPlanToken};
    workspace = {gradient_scratch.get(),
                 static_cast<std::int64_t>(coordinates),
                 force_scratch.get(),
                 static_cast<std::int64_t>(coordinates),
                 coordination_adjoints.get(),
                 static_cast<std::int64_t>(atoms),
                 selected.get(),
                 static_cast<std::int64_t>(batch),
                 primitive_system_errors.get(),
                 static_cast<std::int64_t>(batch),
                 primitive_device_error.get(),
                 1,
                 sequence_active.get(),
                 1,
                 aes2_workspace,
                 d4_workspace,
                 geometry_workspace,
                 kPlanToken};
    activity = {requested.get(), statuses.get(), static_cast<std::int64_t>(batch), kPlanToken};

    int device_id = -1;
    std::string parameter_error;
    if (cudaGetDevice(&device_id) != cudaSuccess ||
        !xtbloom::detail::ensure_cuda_gfn2_parameters(device_id, parameter_error) ||
        reset_gfn2_es2_device_error_cuda(primitive_device_error.get(), stream) != cudaSuccess ||
        update_gfn2_es2_geometry_cache_cuda(es2_batch, positions.get(), es2_cache, es2_workspace,
                                            primitive_device_error.get(), stream) != cudaSuccess ||
        reset_gfn2_aes2_device_errors_cuda(static_cast<std::int64_t>(batch),
                                           primitive_system_errors.get(),
                                           primitive_device_error.get(), stream) != cudaSuccess ||
        update_gfn2_aes2_geometry_cache_cuda(
            aes2_batch, positions.get(), coordination.get(), aes2_cache, aes2_workspace,
            primitive_system_errors.get(), primitive_device_error.get(), stream) != cudaSuccess ||
        reset_gfn2_geometry_device_errors_cuda(
            static_cast<std::int64_t>(batch), primitive_system_errors.get(),
            primitive_device_error.get(), stream) != cudaSuccess ||
        update_gfn2_geometry_cache_cuda(
            geometry_batch, positions.get(), kGeneration, geometry_cache, geometry_workspace,
            primitive_system_errors.get(), primitive_device_error.get(), stream) != cudaSuccess) {
      return false;
    }
    return cudaStreamSynchronize(stream) == cudaSuccess;
  }

  bool set_force_seed(const std::vector<double>& values, cudaStream_t stream) {
    return forces.copy_from(values.data(), values.size(), stream);
  }

  bool run(std::uint32_t mask, cudaStream_t stream) {
    plan.enabled_components = mask;
    return reset_gfn2_classical_force_device_errors_cuda(
               plan.batch_size, system_errors.get(), device_error.get(), stream) == cudaSuccess &&
           add_gfn2_classical_forces_cuda(plan, activity, input, output, workspace,
                                          system_errors.get(), device_error.get(),
                                          stream) == cudaSuccess;
  }
};

int test_component_and_batch_parity(std::size_t batch_size) {
  std::string error;
  HostFixture host;
  CHECK(host.initialize(batch_size, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CHECK(device.initialize(host, stream));

  const std::uint32_t masks[] = {
      static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kRepulsion),
      static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kES2),
      static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kAES2),
      static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4TwoBody),
      static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4ATM),
      kGfn2ClassicalForceAllComponents,
  };
  for (std::uint32_t mask : masks) {
    std::vector<double> expected;
    CHECK(host.expected_forces(mask, expected, error));
    CHECK(device.set_force_seed(host.force_seed, stream));
    CHECK(device.run(mask, stream));
    std::vector<double> actual(expected.size());
    std::vector<std::uint32_t> system_errors(batch_size);
    std::uint32_t device_error = 1u;
    CHECK(device.forces.copy_to(actual.data(), actual.size(), stream));
    CHECK(device.system_errors.copy_to(system_errors.data(), system_errors.size(), stream));
    CHECK(device.device_error.copy_to(&device_error, 1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(device_error == 0u);
    CHECK(std::all_of(system_errors.begin(), system_errors.end(),
                      [](std::uint32_t value) { return value == 0u; }));
    for (std::size_t index = 0; index < actual.size(); ++index) {
      CHECK(near(actual[index], expected[index],
                 mask == kGfn2ClassicalForceAllComponents ? 8.0e-10 : 4.0e-10));
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_terminal_gate_and_peer_failure() {
  std::string error;
  HostFixture host;
  CHECK(host.initialize(8u, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CHECK(device.initialize(host, stream));
  std::vector<double> expected;
  CHECK(host.expected_forces(kGfn2ClassicalForceAllComponents, expected, error));

  std::vector<std::uint8_t> requested(8u, 1u);
  std::vector<xtbloom_status_t> statuses(8u, XTBLOOM_STATUS_SUCCESS);
  requested[1] = 0u;
  statuses[2] = XTBLOOM_STATUS_INTERNAL_ERROR;
  CHECK(device.requested.copy_from(requested.data(), requested.size(), stream));
  CHECK(device.statuses.copy_from(statuses.data(), statuses.size(), stream));
  std::vector<double> poisoned_positions = host.positions;
  std::vector<double> poisoned_shells = host.shell_charges;
  std::vector<double> poisoned_charges = host.atomic_charges;
  for (std::size_t system : {1u, 2u}) {
    const std::size_t atom = static_cast<std::size_t>(host.atom_offsets[system]);
    poisoned_positions[atom * 3u] = std::numeric_limits<double>::quiet_NaN();
    const std::size_t shell = static_cast<std::size_t>(host.es2_plan.batch_shell_offsets()[system]);
    poisoned_shells[shell] = std::numeric_limits<double>::quiet_NaN();
    poisoned_charges[atom] = std::numeric_limits<double>::quiet_NaN();
  }
  CHECK(device.positions.copy_from(poisoned_positions.data(), poisoned_positions.size(), stream));
  CHECK(device.shell_charges.copy_from(poisoned_shells.data(), poisoned_shells.size(), stream));
  CHECK(device.atomic_charges.copy_from(poisoned_charges.data(), poisoned_charges.size(), stream));
  CHECK(device.set_force_seed(host.force_seed, stream));
  CHECK(device.run(kGfn2ClassicalForceAllComponents, stream));
  std::vector<double> actual(host.force_seed.size());
  CHECK(device.forces.copy_to(actual.data(), actual.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (std::size_t system = 0; system < 8u; ++system) {
    const std::size_t begin = static_cast<std::size_t>(host.atom_offsets[system]) * 3u;
    const std::size_t end = static_cast<std::size_t>(host.atom_offsets[system + 1u]) * 3u;
    for (std::size_t coordinate = begin; coordinate < end; ++coordinate) {
      const double reference =
          system == 1u || system == 2u ? host.force_seed[coordinate] : expected[coordinate];
      CHECK(near(actual[coordinate], reference, 8.0e-10));
    }
  }

  /* A requested SUCCESS peer with poisoned q fails locally without rolling back peers. */
  statuses.assign(8u, XTBLOOM_STATUS_SUCCESS);
  requested.assign(8u, 1u);
  poisoned_positions = host.positions;
  poisoned_shells = host.shell_charges;
  poisoned_charges = host.atomic_charges;
  poisoned_charges[static_cast<std::size_t>(host.atom_offsets[3])] =
      std::numeric_limits<double>::quiet_NaN();
  CHECK(device.requested.copy_from(requested.data(), requested.size(), stream));
  CHECK(device.statuses.copy_from(statuses.data(), statuses.size(), stream));
  CHECK(device.positions.copy_from(poisoned_positions.data(), poisoned_positions.size(), stream));
  CHECK(device.shell_charges.copy_from(poisoned_shells.data(), poisoned_shells.size(), stream));
  CHECK(device.atomic_charges.copy_from(poisoned_charges.data(), poisoned_charges.size(), stream));
  CHECK(device.set_force_seed(host.force_seed, stream));
  CHECK(device.run(kGfn2ClassicalForceAllComponents, stream));
  std::vector<std::uint32_t> system_errors(8u);
  CHECK(device.forces.copy_to(actual.data(), actual.size(), stream));
  CHECK(device.system_errors.copy_to(system_errors.data(), system_errors.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(system_errors[3] ==
        static_cast<std::uint32_t>(Gfn2ClassicalForceDeviceError::kAES2Failure));
  for (std::size_t system = 0; system < 8u; ++system) {
    const std::size_t begin = static_cast<std::size_t>(host.atom_offsets[system]) * 3u;
    const std::size_t end = static_cast<std::size_t>(host.atom_offsets[system + 1u]) * 3u;
    for (std::size_t coordinate = begin; coordinate < end; ++coordinate) {
      const double reference = system == 3u ? host.force_seed[coordinate] : expected[coordinate];
      CHECK(near(actual[coordinate], reference, 8.0e-10));
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_combined_finite_difference_and_invariance() {
  std::string error;
  HostFixture host;
  CHECK(host.initialize(1u, error));
  std::vector<double> expected;
  CHECK(host.expected_forces(kGfn2ClassicalForceAllComponents, expected, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CHECK(device.initialize(host, stream));
  CHECK(device.set_force_seed(host.force_seed, stream));
  CHECK(device.run(kGfn2ClassicalForceAllComponents, stream));
  std::vector<double> actual(expected.size());
  CHECK(device.forces.copy_to(actual.data(), actual.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  constexpr double step = 2.0e-5;
  for (std::size_t coordinate = 0; coordinate < host.positions.size(); ++coordinate) {
    std::vector<double> left = host.positions;
    std::vector<double> right = host.positions;
    left[coordinate] -= step;
    right[coordinate] += step;
    double left_energy = 0.0;
    double right_energy = 0.0;
    CHECK(host.total_energy_at(left, left_energy, error));
    CHECK(host.total_energy_at(right, right_energy, error));
    const double numerical_force = -(right_energy - left_energy) / (2.0 * step);
    const double cuda_contribution = actual[coordinate] - host.force_seed[coordinate];
    CHECK(near(cuda_contribution, numerical_force, 2.0e-6));
  }

  for (int axis = 0; axis < 3; ++axis) {
    double net_force = 0.0;
    for (std::size_t atom = 0; atom < host.atomic_numbers.size(); ++atom) {
      net_force += actual[atom * 3u + static_cast<std::size_t>(axis)] -
                   host.force_seed[atom * 3u + static_cast<std::size_t>(axis)];
    }
    CHECK(std::abs(net_force) < 2.0e-10);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_combined_rotation_covariance() {
  std::string error;
  HostFixture original;
  CHECK(original.initialize(1u, error));
  std::fill(original.force_seed.begin(), original.force_seed.end(), 0.0);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture original_device;
  CHECK(original_device.initialize(original, stream));
  CHECK(original_device.set_force_seed(original.force_seed, stream));
  CHECK(original_device.run(kGfn2ClassicalForceAllComponents, stream));
  std::vector<double> original_force(original.force_seed.size());
  CHECK(original_device.forces.copy_to(original_force.data(), original_force.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  HostFixture rotated;
  CHECK(rotated.initialize(1u, error));
  const Rotation rotation = make_test_rotation();
  for (std::size_t atom = 0; atom < rotated.atomic_numbers.size(); ++atom) {
    const std::array<double, 3> position =
        rotate_vector(rotation, rotated.positions.data() + atom * 3u);
    const std::array<double, 3> dipole =
        rotate_vector(rotation, rotated.dipoles.data() + atom * 3u);
    const std::array<double, 6> quadrupole =
        rotate_quadrupole(rotation, rotated.quadrupoles.data() + atom * 6u);
    std::copy(position.begin(), position.end(), rotated.positions.begin() + atom * 3u);
    std::copy(dipole.begin(), dipole.end(), rotated.dipoles.begin() + atom * 3u);
    std::copy(quadrupole.begin(), quadrupole.end(), rotated.quadrupoles.begin() + atom * 6u);
  }
  std::fill(rotated.force_seed.begin(), rotated.force_seed.end(), 0.0);
  CHECK(rotated.refresh_geometry(error));

  DeviceFixture rotated_device;
  CHECK(rotated_device.initialize(rotated, stream));
  CHECK(rotated_device.set_force_seed(rotated.force_seed, stream));
  CHECK(rotated_device.run(kGfn2ClassicalForceAllComponents, stream));
  std::vector<double> rotated_force(rotated.force_seed.size());
  CHECK(rotated_device.forces.copy_to(rotated_force.data(), rotated_force.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (std::size_t atom = 0; atom < original.atomic_numbers.size(); ++atom) {
    const std::array<double, 3> expected =
        rotate_vector(rotation, original_force.data() + atom * 3u);
    for (std::size_t axis = 0; axis < 3u; ++axis) {
      CHECK(near(rotated_force[atom * 3u + axis], expected[axis], 2.0e-9));
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_alias_validation() {
  std::string error;
  HostFixture host;
  CHECK(host.initialize(1u, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CHECK(device.initialize(host, stream));

  Gfn2ClassicalForceDeviceInput bad_input = device.input;
  bad_input.atomic_dipoles = device.output.forces;
  CHECK(add_gfn2_classical_forces_cuda(device.plan, device.activity, bad_input, device.output,
                                       device.workspace, device.system_errors.get(),
                                       device.device_error.get(), stream) == cudaErrorInvalidValue);

  bad_input = device.input;
  bad_input.atomic_quadrupoles = device.output.forces;
  CHECK(add_gfn2_classical_forces_cuda(device.plan, device.activity, bad_input, device.output,
                                       device.workspace, device.system_errors.get(),
                                       device.device_error.get(), stream) == cudaErrorInvalidValue);

  Gfn2ClassicalForceDeviceWorkspace bad_workspace = device.workspace;
  bad_workspace.aes2_workspace.gradient_scratch = device.output.forces;
  CHECK(add_gfn2_classical_forces_cuda(device.plan, device.activity, device.input, device.output,
                                       bad_workspace, device.system_errors.get(),
                                       device.device_error.get(), stream) == cudaErrorInvalidValue);

  bad_workspace = device.workspace;
  bad_workspace.geometry_workspace.gradient_scratch = device.workspace.force_scratch;
  CHECK(add_gfn2_classical_forces_cuda(device.plan, device.activity, device.input, device.output,
                                       bad_workspace, device.system_errors.get(),
                                       device.device_error.get(), stream) == cudaErrorInvalidValue);

  bad_workspace = device.workspace;
  bad_workspace.d4_workspace.gradient_scratch = device.workspace.force_scratch;
  CHECK(add_gfn2_classical_forces_cuda(device.plan, device.activity, device.input, device.output,
                                       bad_workspace, device.system_errors.get(),
                                       device.device_error.get(), stream) == cudaErrorInvalidValue);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_plan_and_activity_failures() {
  std::string error;
  HostFixture host;
  CHECK(host.initialize(8u, error));
  std::vector<double> expected;
  CHECK(host.expected_forces(kGfn2ClassicalForceAllComponents, expected, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CHECK(device.initialize(host, stream));

  std::vector<std::uint8_t> requested(8u, 1u);
  requested[4] = 2u;
  CHECK(device.requested.copy_from(requested.data(), requested.size(), stream));
  CHECK(device.set_force_seed(host.force_seed, stream));
  CHECK(device.run(kGfn2ClassicalForceAllComponents, stream));
  std::vector<double> actual(expected.size());
  std::vector<std::uint32_t> system_errors(8u);
  CHECK(device.forces.copy_to(actual.data(), actual.size(), stream));
  CHECK(device.system_errors.copy_to(system_errors.data(), system_errors.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(system_errors[4] ==
        static_cast<std::uint32_t>(Gfn2ClassicalForceDeviceError::kInvalidActivity));
  for (std::size_t system = 0; system < 8u; ++system) {
    const std::size_t begin = static_cast<std::size_t>(host.atom_offsets[system]) * 3u;
    const std::size_t end = static_cast<std::size_t>(host.atom_offsets[system + 1u]) * 3u;
    for (std::size_t coordinate = begin; coordinate < end; ++coordinate) {
      CHECK(near(actual[coordinate],
                 system == 4u ? host.force_seed[coordinate] : expected[coordinate], 8.0e-10));
    }
  }

  /* A malformed immutable pair partition is plan-wide and suppresses all publication. */
  requested.assign(8u, 1u);
  CHECK(device.requested.copy_from(requested.data(), requested.size(), stream));
  std::vector<std::int64_t> bad_pairs = host.aes2_plan.pair_offsets();
  ++bad_pairs[1];
  CHECK(device.pair_offsets.copy_from(bad_pairs.data(), bad_pairs.size(), stream));
  CHECK(device.set_force_seed(host.force_seed, stream));
  CHECK(device.run(static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4ATM), stream));
  std::uint32_t device_error = 0u;
  CHECK(device.forces.copy_to(actual.data(), actual.size(), stream));
  CHECK(device.device_error.copy_to(&device_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(device_error ==
        static_cast<std::uint32_t>(Gfn2ClassicalForceDeviceError::kInvalidTopology));
  for (std::size_t coordinate = 0; coordinate < actual.size(); ++coordinate) {
    CHECK(actual[coordinate] == host.force_seed[coordinate]);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_graph_replay() {
  std::string error;
  HostFixture host;
  CHECK(host.initialize(8u, error));
  std::vector<double> expected;
  CHECK(host.expected_forces(kGfn2ClassicalForceAllComponents, expected, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CHECK(device.initialize(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  device.plan.enabled_components = kGfn2ClassicalForceAllComponents;

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CHECK(device.forces.copy_device_from(device.force_seed, host.force_seed.size(), stream));
  CUDA_CHECK(reset_gfn2_classical_force_device_errors_cuda(
      device.plan.batch_size, device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(add_gfn2_classical_forces_cuda(
      device.plan, device.activity, device.input, device.output, device.workspace,
      device.system_errors.get(), device.device_error.get(), stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0u));
  for (int replay = 0; replay < 2; ++replay) {
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  std::vector<double> actual(expected.size());
  CHECK(device.forces.copy_to(actual.data(), actual.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (std::size_t index = 0; index < actual.size(); ++index) {
    CHECK(near(actual[index], expected[index], 8.0e-10));
  }
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main() {
  for (std::size_t batch : {1u, 8u, 32u, 128u}) {
    if (const int line = test_component_and_batch_parity(batch); line != 0) {
      return line;
    }
  }
  if (const int line = test_terminal_gate_and_peer_failure(); line != 0) {
    return line;
  }
  if (const int line = test_combined_finite_difference_and_invariance(); line != 0) {
    return line;
  }
  if (const int line = test_combined_rotation_covariance(); line != 0) {
    return line;
  }
  if (const int line = test_alias_validation(); line != 0) {
    return line;
  }
  if (const int line = test_plan_and_activity_failures(); line != 0) {
    return line;
  }
  if (const int line = test_graph_replay(); line != 0) {
    return line;
  }
  return 0;
}
