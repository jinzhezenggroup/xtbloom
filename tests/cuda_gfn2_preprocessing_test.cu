#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_preprocessing.cuh"
#include "model/gfn2/aes2.hpp"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/coordination.hpp"
#include "model/gfn2/es2.hpp"
#include "model/gfn2/h0.hpp"
#include "model/gfn2/integrals.hpp"

namespace {

using namespace gpuxtb::detail;
using namespace gpuxtb::detail::cuda;
using namespace gpuxtb::detail::gfn2;

#define CHECK(condition)                                                                    \
  do {                                                                                      \
    if (!(condition)) {                                                                     \
      std::fprintf(stderr, "preprocessing check failed at %s:%d: %s\n", __FILE__, __LINE__, \
                   #condition);                                                             \
      return __LINE__;                                                                      \
    }                                                                                       \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

constexpr std::uint64_t kPlanToken = 0x120113116ULL;
constexpr double kSentinel = -8123.75;

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
    return count == 0u ? cudaSuccess
                       : cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
  }
  cudaError_t upload(const T* source, std::size_t count, cudaStream_t stream = nullptr) {
    if (count > count_ || (count != 0u && source == nullptr)) return cudaErrorInvalidValue;
    return count == 0u
               ? cudaSuccess
               : cudaMemcpyAsync(data_, source, count * sizeof(T), cudaMemcpyHostToDevice, stream);
  }
  cudaError_t download(T* target, std::size_t count, cudaStream_t stream = nullptr) const {
    if (count > count_ || (count != 0u && target == nullptr)) return cudaErrorInvalidValue;
    return count == 0u
               ? cudaSuccess
               : cudaMemcpyAsync(target, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
  }
  cudaError_t fill(T value, cudaStream_t stream = nullptr) {
    std::vector<T> host(count_, value);
    return upload(host.data(), host.size(), stream);
  }
  T* get() noexcept { return data_; }
  const T* get() const noexcept { return data_; }
  std::size_t size() const noexcept { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

template <typename T>
cudaError_t allocate_upload(DeviceBuffer<T>& target, const std::vector<T>& source,
                            cudaStream_t stream) {
  cudaError_t status = target.allocate(source.size());
  return status == cudaSuccess ? target.upload(source.data(), source.size(), stream) : status;
}

bool near(double actual, double expected, double tolerance = 8.0e-11) {
  return std::abs(actual - expected) <=
         tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

bool vectors_near(const std::vector<double>& actual, const std::vector<double>& expected,
                  double tolerance = 8.0e-11) {
  if (actual.size() != expected.size()) return false;
  for (std::size_t index = 0u; index < actual.size(); ++index) {
    if (!near(actual[index], expected[index], tolerance)) {
      std::fprintf(stderr, "mismatch[%zu] actual=%.17g expected=%.17g\n", index, actual[index],
                   expected[index]);
      return false;
    }
  }
  return true;
}

double logistic(double argument) {
  if (argument >= 0.0) {
    const double exponential = std::exp(-argument);
    return 1.0 / (1.0 + exponential);
  }
  const double exponential = std::exp(argument);
  return exponential / (1.0 + exponential);
}

struct Reference {
  std::vector<double> pair_data;
  std::vector<double> coordination;
  std::vector<double> overlap;
  std::vector<double> dipole;
  std::vector<double> quadrupole;
  std::vector<double> h0;
  std::vector<double> es2;
  std::vector<double> aes2;
};

struct HostCase {
  std::int64_t batch = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> pair_offsets;
  std::vector<std::int64_t> shell_pair_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::int64_t maximum_system_shells = 0;
  CoordinationPlan coordination_plan;
  BasisPlan basis;
  IntegralPlan integrals;
  H0Plan h0_plan;
  ES2Plan es2_plan;
  AES2Plan aes2_plan;

  bool create(std::int64_t batch_size, std::string& error) {
    batch = batch_size;
    atom_offsets.resize(static_cast<std::size_t>(batch + 1));
    pair_offsets.resize(static_cast<std::size_t>(batch + 1));
    for (std::int64_t system = 0; system < batch; ++system) {
      atom_offsets[static_cast<std::size_t>(system)] = 2 * system;
      pair_offsets[static_cast<std::size_t>(system)] = system;
      const std::int32_t first_elements[4]{6, 8, 3, 1};
      atomic_numbers.push_back(first_elements[static_cast<std::size_t>(system % 4)]);
      atomic_numbers.push_back(1);
      const double shift = 6.0 * static_cast<double>(system);
      const double stretch = 1.35 + 0.01 * static_cast<double>(system % 11);
      positions.insert(positions.end(),
                       {shift - 0.2, 0.05 * static_cast<double>(system % 3),
                        -0.03 * static_cast<double>(system % 5), shift + stretch, 0.31, -0.17});
    }
    atom_offsets.back() = 2 * batch;
    pair_offsets.back() = batch;
    const std::int64_t atoms = static_cast<std::int64_t>(atomic_numbers.size());
    if (make_coordination_plan(batch, atoms, atom_offsets.data(), atomic_numbers.data(),
                               coordination_plan, error) != GPUXTB_STATUS_SUCCESS ||
        make_basis_plan(batch, atoms, atom_offsets.data(), atomic_numbers.data(), basis, error) !=
            GPUXTB_STATUS_SUCCESS ||
        make_integral_plan(basis, integrals, error) != GPUXTB_STATUS_SUCCESS ||
        make_h0_plan(basis, integrals, atomic_numbers.data(), h0_plan, error) !=
            GPUXTB_STATUS_SUCCESS ||
        make_es2_plan(basis, atomic_numbers.data(), es2_plan, error) != GPUXTB_STATUS_SUCCESS ||
        make_aes2_plan(basis, atomic_numbers.data(), aes2_plan, error) != GPUXTB_STATUS_SUCCESS) {
      return false;
    }
    shell_pair_offsets.assign(static_cast<std::size_t>(batch + 1), 0);
    for (std::int64_t system = 0; system < batch; ++system) {
      const std::int64_t shells = basis.batch_shell_offsets[static_cast<std::size_t>(system + 1)] -
                                  basis.batch_shell_offsets[static_cast<std::size_t>(system)];
      maximum_system_shells = std::max(maximum_system_shells, shells);
      shell_pair_offsets[static_cast<std::size_t>(system + 1)] =
          shell_pair_offsets[static_cast<std::size_t>(system)] + shells * shells;
    }
    return shell_pair_offsets == h0_plan.shell_pair_offsets &&
           pair_offsets == aes2_plan.pair_offsets();
  }

  bool evaluate(const std::vector<double>& coordinates, std::uint64_t generation, Reference& result,
                std::string& error) const {
    const std::size_t atoms = static_cast<std::size_t>(basis.total_atoms);
    const std::size_t matrices = static_cast<std::size_t>(integrals.total_matrix_elements);
    result.coordination.resize(atoms);
    result.pair_data.resize(static_cast<std::size_t>(pair_offsets.back()) *
                            static_cast<std::size_t>(kGfn2GeometryPairDataElements));
    result.overlap.resize(matrices);
    result.dipole.resize(3u * matrices);
    result.quadrupole.resize(6u * matrices);
    result.h0.resize(matrices);
    result.es2.resize(static_cast<std::size_t>(es2_plan.total_matrix_elements()));
    result.aes2.resize(static_cast<std::size_t>(aes2_plan.pair_data_elements()));
    if (evaluate_coordination_cpu(coordination_plan, coordinates.data(), result.coordination.data(),
                                  error) != GPUXTB_STATUS_SUCCESS) {
      return false;
    }
    for (std::int64_t system = 0; system < batch; ++system) {
      const std::int64_t lower = atom_offsets[static_cast<std::size_t>(system)];
      const std::int64_t upper = lower + 1;
      const double dx = coordinates[static_cast<std::size_t>(upper * 3)] -
                        coordinates[static_cast<std::size_t>(lower * 3)];
      const double dy = coordinates[static_cast<std::size_t>(upper * 3 + 1)] -
                        coordinates[static_cast<std::size_t>(lower * 3 + 1)];
      const double dz = coordinates[static_cast<std::size_t>(upper * 3 + 2)] -
                        coordinates[static_cast<std::size_t>(lower * 3 + 2)];
      const double distance = std::hypot(std::hypot(dx, dy), dz);
      const double inverse = 1.0 / distance;
      const double radius = coordination_plan.covalent_radius[static_cast<std::size_t>(lower)] +
                            coordination_plan.covalent_radius[static_cast<std::size_t>(upper)];
      const double first = logistic(10.0 * (radius * inverse - 1.0));
      const double second = logistic(20.0 * ((radius + 2.0) * inverse - 1.0));
      const double count = first * second;
      const double derivative = -inverse * inverse *
                                (10.0 * radius * first * (1.0 - first) * second +
                                 20.0 * (radius + 2.0) * second * (1.0 - second) * first);
      const std::size_t base = static_cast<std::size_t>(system) *
                               static_cast<std::size_t>(kGfn2GeometryPairDataElements);
      result.pair_data[base] = dx;
      result.pair_data[base + 1u] = dy;
      result.pair_data[base + 2u] = dz;
      result.pair_data[base + 3u] = distance;
      result.pair_data[base + 4u] = inverse;
      result.pair_data[base + 5u] = count;
      result.pair_data[base + 6u] = derivative * inverse;
    }
    std::vector<double> integral_workspace((integrals.workspace_size_bytes + sizeof(double) - 1u) /
                                           sizeof(double));
    if (evaluate_overlap_cpu(basis, integrals, coordinates.data(), result.overlap.data(),
                             integral_workspace.data(), integral_workspace.size() * sizeof(double),
                             error) != GPUXTB_STATUS_SUCCESS ||
        evaluate_multipole_cpu(basis, integrals, coordinates.data(), result.dipole.data(),
                               result.quadrupole.data(), integral_workspace.data(),
                               integral_workspace.size() * sizeof(double),
                               error) != GPUXTB_STATUS_SUCCESS ||
        evaluate_h0_cpu(basis, integrals, h0_plan, coordinates.data(), result.coordination.data(),
                        result.overlap.data(), result.h0.data(), error) != GPUXTB_STATUS_SUCCESS) {
      return false;
    }
    std::vector<double> es2_scratch(result.es2.size());
    ES2Workspace es2_workspace{};
    es2_workspace.matrix_scratch = es2_scratch.data();
    es2_workspace.matrix_elements = es2_plan.total_matrix_elements();
    ES2GeometryCache es2_cache{};
    if (update_es2_geometry_cache_cpu(es2_plan, coordinates.data(), generation, result.es2.data(),
                                      result.es2.size(), es2_workspace, es2_cache,
                                      error) != GPUXTB_STATUS_SUCCESS) {
      return false;
    }
    std::vector<double> aes2_scratch(result.aes2.size());
    AES2Workspace aes2_workspace{};
    aes2_workspace.pair_scratch = aes2_scratch.data();
    aes2_workspace.pair_elements = aes2_plan.pair_data_elements();
    AES2GeometryCache aes2_cache{};
    return update_aes2_geometry_cache_cpu(aes2_plan, coordinates.data(), result.coordination.data(),
                                          generation, result.aes2.data(), result.aes2.size(),
                                          aes2_workspace, aes2_cache,
                                          error) == GPUXTB_STATUS_SUCCESS;
  }
};

struct DeviceFixture {
  DeviceBuffer<std::int64_t> atom_offsets, pair_offsets, batch_shell_offsets, batch_orbital_offsets,
      matrix_offsets, shell_pair_offsets, atom_shell_offsets, shell_orbital_offsets,
      shell_primitive_offsets, shell_to_atom, es2_matrix_offsets;
  DeviceBuffer<std::uint8_t> angular_momenta, requested, published;
  DeviceBuffer<double> covalent_radii, primitive_exponents, primitive_coefficients, atomic_radii,
      shell_levels, shell_coordination_scale, shell_polynomial, shell_pair_scale, es2_hardness,
      aes2_dipole_kernel, aes2_quadrupole_kernel, aes2_radius, aes2_valence, positions;

  DeviceBuffer<double> public_geometry_pair, public_coordination, public_overlap, public_dipole,
      public_quadrupole, public_h0, public_es2, public_aes2;
  DeviceBuffer<std::uint64_t> public_geometry_generation, public_operator_generation;

  DeviceBuffer<double> position_scratch, candidate_geometry_pair, candidate_coordination,
      geometry_pair_scratch, geometry_coordination_scratch, candidate_overlap, candidate_dipole,
      candidate_quadrupole, candidate_h0, integral_overlap_scratch, integral_dipole_scratch,
      integral_quadrupole_scratch, integral_h0_scratch, candidate_es2, es2_scratch, candidate_aes2,
      aes2_scratch;
  DeviceBuffer<std::uint64_t> candidate_geometry_generation;
  DeviceBuffer<std::uint32_t> geometry_sequence, integral_sequence, geometry_system_errors,
      geometry_device_error, integral_system_errors, integral_device_error, es2_device_error,
      aes2_system_errors, aes2_device_error, system_stages, plan_error;

  Gfn2PreprocessingDeviceBinding binding{};

  cudaError_t initialize(const HostCase& host, cudaStream_t stream) {
    cudaError_t status = allocate_upload(atom_offsets, host.basis.atom_offsets, stream);
#define UPLOAD(field, source) \
  if (status == cudaSuccess) status = allocate_upload(field, source, stream)
    UPLOAD(pair_offsets, host.pair_offsets);
    UPLOAD(batch_shell_offsets, host.basis.batch_shell_offsets);
    UPLOAD(batch_orbital_offsets, host.basis.batch_orbital_offsets);
    UPLOAD(matrix_offsets, host.integrals.matrix_offsets);
    UPLOAD(shell_pair_offsets, host.shell_pair_offsets);
    UPLOAD(atom_shell_offsets, host.basis.atom_shell_offsets);
    UPLOAD(shell_orbital_offsets, host.basis.shell_orbital_offsets);
    UPLOAD(shell_primitive_offsets, host.basis.shell_primitive_offsets);
    UPLOAD(shell_to_atom, host.basis.shell_to_atom);
    UPLOAD(angular_momenta, host.basis.angular_momenta);
    UPLOAD(primitive_exponents, host.basis.primitive_exponents);
    UPLOAD(primitive_coefficients, host.basis.primitive_coefficients);
    UPLOAD(covalent_radii, host.coordination_plan.covalent_radius);
    UPLOAD(atomic_radii, host.h0_plan.atomic_radii);
    UPLOAD(shell_levels, host.h0_plan.shell_levels);
    UPLOAD(shell_coordination_scale, host.h0_plan.shell_coordination_scale);
    UPLOAD(shell_polynomial, host.h0_plan.shell_polynomial);
    UPLOAD(shell_pair_scale, host.h0_plan.shell_pair_scale);
    UPLOAD(es2_matrix_offsets, host.es2_plan.matrix_offsets());
    UPLOAD(es2_hardness, host.es2_plan.shell_hardness());
    UPLOAD(aes2_dipole_kernel, host.aes2_plan.dipole_kernel());
    UPLOAD(aes2_quadrupole_kernel, host.aes2_plan.quadrupole_kernel());
    UPLOAD(aes2_radius, host.aes2_plan.multipole_radius());
    UPLOAD(aes2_valence, host.aes2_plan.multipole_valence_cn());
    UPLOAD(positions, host.positions);
#undef UPLOAD
    const std::size_t batch = static_cast<std::size_t>(host.batch);
    const std::size_t atoms = static_cast<std::size_t>(host.basis.total_atoms);
    const std::size_t pairs = static_cast<std::size_t>(host.pair_offsets.back());
    const std::size_t matrices = static_cast<std::size_t>(host.integrals.total_matrix_elements);
    const std::size_t geometry_pairs =
        pairs * static_cast<std::size_t>(kGfn2GeometryPairDataElements);
    const std::size_t aes2_pairs = pairs * static_cast<std::size_t>(kGfn2AES2PairDataElements);
    const std::size_t es2_matrices =
        static_cast<std::size_t>(host.es2_plan.total_matrix_elements());
#define ALLOC(field, count) \
  if (status == cudaSuccess) status = field.allocate(count)
    ALLOC(requested, batch);
    ALLOC(published, batch);
    ALLOC(public_geometry_pair, geometry_pairs);
    ALLOC(public_coordination, atoms);
    ALLOC(public_geometry_generation, batch);
    ALLOC(public_overlap, matrices);
    ALLOC(public_dipole, 3u * matrices);
    ALLOC(public_quadrupole, 6u * matrices);
    ALLOC(public_h0, matrices);
    ALLOC(public_es2, es2_matrices);
    ALLOC(public_aes2, aes2_pairs);
    ALLOC(public_operator_generation, batch);
    ALLOC(position_scratch, 3u * atoms);
    ALLOC(candidate_geometry_pair, geometry_pairs);
    ALLOC(candidate_coordination, atoms);
    ALLOC(candidate_geometry_generation, batch);
    ALLOC(geometry_pair_scratch, geometry_pairs);
    ALLOC(geometry_coordination_scratch, atoms);
    ALLOC(geometry_sequence, 1u);
    ALLOC(candidate_overlap, matrices);
    ALLOC(candidate_dipole, 3u * matrices);
    ALLOC(candidate_quadrupole, 6u * matrices);
    ALLOC(candidate_h0, matrices);
    ALLOC(integral_overlap_scratch, matrices);
    ALLOC(integral_dipole_scratch, 3u * matrices);
    ALLOC(integral_quadrupole_scratch, 6u * matrices);
    ALLOC(integral_h0_scratch, matrices);
    ALLOC(integral_sequence, 1u);
    ALLOC(candidate_es2, es2_matrices);
    ALLOC(es2_scratch, es2_matrices);
    ALLOC(candidate_aes2, aes2_pairs);
    ALLOC(aes2_scratch, aes2_pairs);
    ALLOC(geometry_system_errors, batch);
    ALLOC(geometry_device_error, 1u);
    ALLOC(integral_system_errors, batch);
    ALLOC(integral_device_error, 1u);
    ALLOC(es2_device_error, 1u);
    ALLOC(aes2_system_errors, batch);
    ALLOC(aes2_device_error, 1u);
    ALLOC(system_stages, batch);
    ALLOC(plan_error, 1u);
#undef ALLOC
    if (status != cudaSuccess) return status;
    std::vector<std::uint8_t> all_active(batch, 1u);
    status = requested.upload(all_active.data(), all_active.size(), stream);
    if (status != cudaSuccess) return status;
    status = seed_public(stream);
    if (status != cudaSuccess) return status;
    bind(host);
    const auto seal = seal_gfn2_preprocessing_binding_cuda(binding);
    return seal.success() ? cudaSuccess : cudaErrorInvalidValue;
  }

  void bind(const HostCase& host) {
    const std::int64_t batch = host.batch;
    const std::int64_t atoms = host.basis.total_atoms;
    const std::int64_t pairs = host.pair_offsets.back();
    const std::int64_t matrices = host.integrals.total_matrix_elements;
    const std::int64_t geometry_pairs = pairs * kGfn2GeometryPairDataElements;
    const std::int64_t aes2_pairs = pairs * kGfn2AES2PairDataElements;
    const std::int64_t es2_matrices = host.es2_plan.total_matrix_elements();
    binding = {};
    binding.plan.abi_version = kGfn2PreprocessingAbiVersion;
    binding.plan.plan_token = kPlanToken;
    binding.plan.geometry = {batch,
                             atoms,
                             pairs,
                             batch + 1,
                             batch + 1,
                             atoms,
                             3 * atoms,
                             kPlanToken,
                             atom_offsets.get(),
                             pair_offsets.get(),
                             covalent_radii.get()};
    binding.plan.integrals = {batch,
                              atoms,
                              host.basis.total_shells,
                              host.basis.total_orbitals,
                              host.basis.total_primitives,
                              matrices,
                              host.shell_pair_offsets.back(),
                              host.maximum_system_shells,
                              host.integrals.integral_cutoff,
                              kPlanToken,
                              batch + 1,
                              batch + 1,
                              batch + 1,
                              batch + 1,
                              batch + 1,
                              atoms + 1,
                              host.basis.total_shells + 1,
                              host.basis.total_shells + 1,
                              host.basis.total_shells,
                              host.basis.total_shells,
                              host.basis.total_primitives,
                              host.basis.total_primitives,
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
    binding.plan.h0 = {atoms,
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
    binding.plan.es2 = {batch,
                        atoms,
                        host.basis.total_shells,
                        es2_matrices,
                        kPlanToken,
                        batch + 1,
                        batch + 1,
                        atoms + 1,
                        batch + 1,
                        host.basis.total_shells,
                        host.basis.total_shells,
                        atom_offsets.get(),
                        batch_shell_offsets.get(),
                        atom_shell_offsets.get(),
                        es2_matrix_offsets.get(),
                        shell_to_atom.get(),
                        es2_hardness.get()};
    binding.plan.aes2 = {batch,
                         atoms,
                         pairs,
                         kPlanToken,
                         batch + 1,
                         batch + 1,
                         atoms,
                         atoms,
                         atoms,
                         atoms,
                         atom_offsets.get(),
                         pair_offsets.get(),
                         aes2_dipole_kernel.get(),
                         aes2_quadrupole_kernel.get(),
                         aes2_radius.get(),
                         aes2_valence.get()};
    binding.input = {positions.get(), 3 * atoms, kPlanToken};
    binding.activity = {requested.get(), batch, published.get(), batch, kPlanToken};
    binding.output.geometry = {public_geometry_pair.get(),
                               geometry_pairs,
                               public_coordination.get(),
                               atoms,
                               public_geometry_generation.get(),
                               batch,
                               kPlanToken};
    binding.output.overlap = public_overlap.get();
    binding.output.overlap_elements = matrices;
    binding.output.dipole_integrals = public_dipole.get();
    binding.output.dipole_elements = 3 * matrices;
    binding.output.quadrupole_integrals = public_quadrupole.get();
    binding.output.quadrupole_elements = 6 * matrices;
    binding.output.h0 = public_h0.get();
    binding.output.h0_elements = matrices;
    binding.output.es2 = {public_es2.get(), es2_matrices, 0u, kPlanToken};
    binding.output.aes2 = {public_aes2.get(), aes2_pairs, 0u, kPlanToken};
    binding.output.operator_generations = public_operator_generation.get();
    binding.output.generation_elements = batch;
    binding.output.plan_token = kPlanToken;
    binding.diagnostics = {geometry_system_errors.get(),
                           batch,
                           geometry_device_error.get(),
                           integral_system_errors.get(),
                           batch,
                           integral_device_error.get(),
                           es2_device_error.get(),
                           aes2_system_errors.get(),
                           batch,
                           aes2_device_error.get(),
                           system_stages.get(),
                           batch,
                           plan_error.get(),
                           kPlanToken};
    binding.workspace.positions_scratch = position_scratch.get();
    binding.workspace.position_elements = 3 * atoms;
    binding.workspace.geometry_candidate = {candidate_geometry_pair.get(),
                                            geometry_pairs,
                                            candidate_coordination.get(),
                                            atoms,
                                            candidate_geometry_generation.get(),
                                            batch,
                                            kPlanToken};
    binding.workspace.geometry = {geometry_pair_scratch.get(),
                                  geometry_pairs,
                                  geometry_coordination_scratch.get(),
                                  atoms,
                                  nullptr,
                                  0,
                                  geometry_sequence.get(),
                                  1,
                                  kPlanToken};
    binding.workspace.overlap_candidate = candidate_overlap.get();
    binding.workspace.overlap_elements = matrices;
    binding.workspace.dipole_candidate = candidate_dipole.get();
    binding.workspace.dipole_elements = 3 * matrices;
    binding.workspace.quadrupole_candidate = candidate_quadrupole.get();
    binding.workspace.quadrupole_elements = 6 * matrices;
    binding.workspace.h0_candidate = candidate_h0.get();
    binding.workspace.h0_elements = matrices;
    binding.workspace.integrals = {integral_overlap_scratch.get(),
                                   matrices,
                                   integral_dipole_scratch.get(),
                                   3 * matrices,
                                   integral_quadrupole_scratch.get(),
                                   6 * matrices,
                                   integral_h0_scratch.get(),
                                   matrices,
                                   integral_sequence.get(),
                                   1,
                                   kPlanToken};
    binding.workspace.es2_candidate = {candidate_es2.get(), es2_matrices, 0u, kPlanToken};
    binding.workspace.es2 = {es2_scratch.get(), es2_matrices, nullptr, 0, nullptr, 0, nullptr, 0};
    binding.workspace.aes2_candidate = {candidate_aes2.get(), aes2_pairs, 0u, kPlanToken};
    binding.workspace.aes2 = {
        aes2_scratch.get(), aes2_pairs, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0};
    binding.workspace.plan_token = kPlanToken;
    binding.plan_token = kPlanToken;
  }

  cudaError_t seed_public(cudaStream_t stream) {
    cudaError_t status = public_geometry_pair.fill(kSentinel, stream);
#define FILL(field) \
  if (status == cudaSuccess) status = field.fill(kSentinel, stream)
    FILL(public_coordination);
    FILL(public_overlap);
    FILL(public_dipole);
    FILL(public_quadrupole);
    FILL(public_h0);
    FILL(public_es2);
    FILL(public_aes2);
#undef FILL
    if (status == cudaSuccess) status = public_geometry_generation.fill(5u, stream);
    if (status == cudaSuccess) status = public_operator_generation.fill(5u, stream);
    if (status == cudaSuccess) status = published.fill(0u, stream);
    return status;
  }

  cudaError_t upload_positions(const std::vector<double>& values, cudaStream_t stream) {
    return positions.upload(values.data(), values.size(), stream);
  }

  cudaError_t upload_activity(const std::vector<std::uint8_t>& values, cudaStream_t stream) {
    return requested.upload(values.data(), values.size(), stream);
  }
};

struct Downloaded {
  std::vector<double> pair_data, coordination, overlap, dipole, quadrupole, h0, es2, aes2;
  std::vector<std::uint64_t> geometry_generations, operator_generations;
  std::vector<std::uint8_t> published;
  std::vector<std::uint32_t> geometry_errors, integral_errors, aes2_errors, stages;
  std::uint32_t plan_error = 0u;
};

cudaError_t download(const HostCase& host, const DeviceFixture& device, Downloaded& values,
                     cudaStream_t stream) {
  values.pair_data.resize(device.public_geometry_pair.size());
  values.coordination.resize(device.public_coordination.size());
  values.overlap.resize(device.public_overlap.size());
  values.dipole.resize(device.public_dipole.size());
  values.quadrupole.resize(device.public_quadrupole.size());
  values.h0.resize(device.public_h0.size());
  values.es2.resize(device.public_es2.size());
  values.aes2.resize(device.public_aes2.size());
  values.geometry_generations.resize(static_cast<std::size_t>(host.batch));
  values.operator_generations.resize(static_cast<std::size_t>(host.batch));
  values.published.resize(static_cast<std::size_t>(host.batch));
  values.geometry_errors.resize(static_cast<std::size_t>(host.batch));
  values.integral_errors.resize(static_cast<std::size_t>(host.batch));
  values.aes2_errors.resize(static_cast<std::size_t>(host.batch));
  values.stages.resize(static_cast<std::size_t>(host.batch));
  cudaError_t status = device.public_geometry_pair.download(values.pair_data.data(),
                                                            values.pair_data.size(), stream);
#define DOWNLOAD(field, target) \
  if (status == cudaSuccess)    \
  status = device.field.download(values.target.data(), values.target.size(), stream)
  DOWNLOAD(public_coordination, coordination);
  DOWNLOAD(public_overlap, overlap);
  DOWNLOAD(public_dipole, dipole);
  DOWNLOAD(public_quadrupole, quadrupole);
  DOWNLOAD(public_h0, h0);
  DOWNLOAD(public_es2, es2);
  DOWNLOAD(public_aes2, aes2);
  DOWNLOAD(public_geometry_generation, geometry_generations);
  DOWNLOAD(public_operator_generation, operator_generations);
  DOWNLOAD(published, published);
  DOWNLOAD(geometry_system_errors, geometry_errors);
  DOWNLOAD(integral_system_errors, integral_errors);
  DOWNLOAD(aes2_system_errors, aes2_errors);
  DOWNLOAD(system_stages, stages);
#undef DOWNLOAD
  if (status == cudaSuccess) status = device.plan_error.download(&values.plan_error, 1u, stream);
  return status;
}

bool matches_reference(const Downloaded& actual, const Reference& expected,
                       std::uint64_t generation) {
  return vectors_near(actual.pair_data, expected.pair_data) &&
         vectors_near(actual.coordination, expected.coordination) &&
         vectors_near(actual.overlap, expected.overlap) &&
         vectors_near(actual.dipole, expected.dipole) &&
         vectors_near(actual.quadrupole, expected.quadrupole) &&
         vectors_near(actual.h0, expected.h0) && vectors_near(actual.es2, expected.es2) &&
         vectors_near(actual.aes2, expected.aes2, 2.0e-10) && actual.plan_error == 0u &&
         std::all_of(actual.geometry_generations.begin(), actual.geometry_generations.end(),
                     [generation](std::uint64_t value) { return value == generation; }) &&
         std::all_of(actual.operator_generations.begin(), actual.operator_generations.end(),
                     [generation](std::uint64_t value) { return value == generation; }) &&
         std::all_of(actual.published.begin(), actual.published.end(),
                     [](std::uint8_t value) { return value == 1u; }) &&
         std::all_of(actual.geometry_errors.begin(), actual.geometry_errors.end(),
                     [](std::uint32_t value) { return value == 0u; }) &&
         std::all_of(actual.integral_errors.begin(), actual.integral_errors.end(),
                     [](std::uint32_t value) { return value == 0u; }) &&
         std::all_of(actual.aes2_errors.begin(), actual.aes2_errors.end(),
                     [](std::uint32_t value) { return value == 0u; });
}

bool stable_binding_except_attempted_generation(
    const Gfn2PreprocessingDeviceBinding& first,
    const Gfn2PreprocessingDeviceBinding& second) {
  Gfn2PreprocessingDeviceBinding normalized_first = first;
  Gfn2PreprocessingDeviceBinding normalized_second = second;
  normalized_first.output.es2.geometry_generation = 0u;
  normalized_first.output.aes2.geometry_generation = 0u;
  normalized_first.workspace.es2_candidate.geometry_generation = 0u;
  normalized_first.workspace.aes2_candidate.geometry_generation = 0u;
  normalized_second.output.es2.geometry_generation = 0u;
  normalized_second.output.aes2.geometry_generation = 0u;
  normalized_second.workspace.es2_candidate.geometry_generation = 0u;
  normalized_second.workspace.aes2_candidate.geometry_generation = 0u;
  return std::memcmp(&normalized_first, &normalized_second,
                     sizeof(normalized_first)) == 0;
}

std::vector<double> changed_positions(const HostCase& host, double scale) {
  std::vector<double> changed = host.positions;
  for (std::int64_t system = 0; system < host.batch; ++system) {
    const std::int64_t upper = host.atom_offsets[static_cast<std::size_t>(system)] + 1;
    changed[static_cast<std::size_t>(upper * 3)] += scale * static_cast<double>(1 + system % 5);
    changed[static_cast<std::size_t>(upper * 3 + 1)] -= 0.5 * scale;
  }
  return changed;
}

int test_cpu_parity_batches() {
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    HostCase host;
    std::string error;
    CHECK(host.create(batch_size, error));
    Reference expected;
    CHECK(host.evaluate(host.positions, 17u, expected, error));
    cudaStream_t stream = nullptr;
    if (batch_size != 1) CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    DeviceFixture device;
    CUDA_CHECK(device.initialize(host, stream));
    const Gfn2PreprocessingDeviceBinding stable_binding = device.binding;
    CHECK(validate_gfn2_preprocessing_binding_cuda(device.binding).success());
    CHECK(compose_gfn2_preprocessing_cuda(device.binding, 17u, stream).success());
    Downloaded actual;
    CUDA_CHECK(download(host, device, actual, stream));
    if (stream == nullptr) {
      CUDA_CHECK(cudaDeviceSynchronize());
    } else {
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CHECK(matches_reference(actual, expected, 17u));

    const std::vector<double> changed = changed_positions(host, 0.007);
    Reference changed_expected;
    CHECK(host.evaluate(changed, 18u, changed_expected, error));
    CUDA_CHECK(device.upload_positions(changed, stream));
    CHECK(compose_gfn2_preprocessing_cuda(device.binding, 18u, stream).success());
    CUDA_CHECK(download(host, device, actual, stream));
    if (stream == nullptr) {
      CUDA_CHECK(cudaDeviceSynchronize());
    } else {
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CHECK(matches_reference(actual, changed_expected, 18u));
    CHECK(stable_binding_except_attempted_generation(stable_binding, device.binding));
    if (stream != nullptr) CUDA_CHECK(cudaStreamDestroy(stream));
  }
  return 0;
}

int test_peer_transaction_and_inactive_mask() {
  HostCase host;
  std::string error;
  CHECK(host.create(8, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CHECK(compose_gfn2_preprocessing_cuda(device.binding, 21u, stream).success());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  Downloaded before;
  CUDA_CHECK(download(host, device, before, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<double> changed = changed_positions(host, 0.013);
  constexpr std::int64_t kFailed = 3;
  constexpr std::int64_t kInactive = 6;
  changed[static_cast<std::size_t>(host.atom_offsets[kFailed] * 3)] =
      std::numeric_limits<double>::quiet_NaN();
  changed[static_cast<std::size_t>(host.atom_offsets[kInactive] * 3)] =
      std::numeric_limits<double>::quiet_NaN();
  std::vector<std::uint8_t> activity(static_cast<std::size_t>(host.batch), 1u);
  activity[static_cast<std::size_t>(kInactive)] = 0u;
  CUDA_CHECK(device.upload_positions(changed, stream));
  CUDA_CHECK(device.upload_activity(activity, stream));
  CHECK(compose_gfn2_preprocessing_cuda(device.binding, 22u, stream).success());
  Downloaded after;
  CUDA_CHECK(download(host, device, after, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  if (after.plan_error != 0u) {
    std::fprintf(stderr, "peer transaction plan_error=%u\n", after.plan_error);
    for (std::size_t system = 0; system < after.stages.size(); ++system) {
      std::fprintf(stderr, "  system=%zu geometry=%u integral=%u aes2=%u stage=%u published=%u\n",
                   system, after.geometry_errors[system], after.integral_errors[system],
                   after.aes2_errors[system], after.stages[system],
                   static_cast<unsigned>(after.published[system]));
    }
  }
  CHECK(after.plan_error == 0u);
  CHECK(after.published[static_cast<std::size_t>(kFailed)] == 0u);
  CHECK(after.published[static_cast<std::size_t>(kInactive)] == 0u);
  CHECK(after.stages[static_cast<std::size_t>(kFailed)] ==
        static_cast<std::uint32_t>(Gfn2PreprocessingSystemStage::kGeometry));
  CHECK(after.stages[static_cast<std::size_t>(kInactive)] == 0u);
  CHECK(after.geometry_generations[static_cast<std::size_t>(kFailed)] == 21u);
  CHECK(after.operator_generations[static_cast<std::size_t>(kFailed)] == 21u);
  CHECK(after.geometry_generations[static_cast<std::size_t>(kInactive)] == 21u);
  CHECK(after.operator_generations[static_cast<std::size_t>(kInactive)] == 21u);

  const auto slice_equal = [](const auto& first, const auto& second, std::int64_t begin,
                              std::int64_t end) {
    return std::equal(first.begin() + begin, first.begin() + end, second.begin() + begin);
  };
  for (const std::int64_t system : {kFailed, kInactive}) {
    const std::int64_t atom_begin = host.atom_offsets[static_cast<std::size_t>(system)];
    const std::int64_t atom_end = host.atom_offsets[static_cast<std::size_t>(system + 1)];
    const std::int64_t matrix_begin =
        host.integrals.matrix_offsets[static_cast<std::size_t>(system)];
    const std::int64_t matrix_end =
        host.integrals.matrix_offsets[static_cast<std::size_t>(system + 1)];
    const std::int64_t es2_begin = host.es2_plan.matrix_offsets()[static_cast<std::size_t>(system)];
    const std::int64_t es2_end =
        host.es2_plan.matrix_offsets()[static_cast<std::size_t>(system + 1)];
    CHECK(slice_equal(before.coordination, after.coordination, atom_begin, atom_end));
    CHECK(slice_equal(before.overlap, after.overlap, matrix_begin, matrix_end));
    for (std::int64_t component = 0; component < kGfn2IntegralDipoleComponents; ++component) {
      const std::int64_t component_begin =
          component * host.integrals.total_matrix_elements + matrix_begin;
      const std::int64_t component_end =
          component * host.integrals.total_matrix_elements + matrix_end;
      CHECK(slice_equal(before.dipole, after.dipole, component_begin, component_end));
    }
    for (std::int64_t component = 0; component < kGfn2IntegralQuadrupoleComponents; ++component) {
      const std::int64_t component_begin =
          component * host.integrals.total_matrix_elements + matrix_begin;
      const std::int64_t component_end =
          component * host.integrals.total_matrix_elements + matrix_end;
      CHECK(slice_equal(before.quadrupole, after.quadrupole, component_begin, component_end));
    }
    CHECK(slice_equal(before.h0, after.h0, matrix_begin, matrix_end));
    CHECK(slice_equal(before.es2, after.es2, es2_begin, es2_end));
    CHECK(slice_equal(before.pair_data, after.pair_data, system * kGfn2GeometryPairDataElements,
                      (system + 1) * kGfn2GeometryPairDataElements));
    CHECK(slice_equal(before.aes2, after.aes2, system * kGfn2AES2PairDataElements,
                      (system + 1) * kGfn2AES2PairDataElements));
  }
  for (std::int64_t system = 0; system < host.batch; ++system) {
    if (system == kFailed || system == kInactive) continue;
    CHECK(after.published[static_cast<std::size_t>(system)] == 1u);
    CHECK(after.geometry_generations[static_cast<std::size_t>(system)] == 22u);
    CHECK(after.operator_generations[static_cast<std::size_t>(system)] == 22u);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_plan_failure_and_seal_fail_closed() {
  HostCase host;
  std::string error;
  CHECK(host.create(8, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));

  Gfn2PreprocessingDeviceBinding atom_overflow = device.binding;
  atom_overflow.plan.geometry.total_atoms = std::numeric_limits<std::int64_t>::max();
  CHECK(seal_gfn2_preprocessing_binding_cuda(atom_overflow).error ==
        Gfn2PreprocessingBindingError::kInvalidExtent);
  Gfn2PreprocessingDeviceBinding shell_overflow = device.binding;
  shell_overflow.plan.integrals.total_shells = std::numeric_limits<std::int64_t>::max();
  CHECK(seal_gfn2_preprocessing_binding_cuda(shell_overflow).error ==
        Gfn2PreprocessingBindingError::kInvalidExtent);

  CHECK(compose_gfn2_preprocessing_cuda(device.binding, 31u, stream).success());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  Downloaded before;
  CUDA_CHECK(download(host, device, before, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const std::int64_t malformed = host.basis.total_atoms + 1;
  CUDA_CHECK(cudaMemcpyAsync(device.atom_offsets.get() + host.batch, &malformed, sizeof(malformed),
                             cudaMemcpyHostToDevice, stream));
  CHECK(compose_gfn2_preprocessing_cuda(device.binding, 32u, stream).success());
  CHECK(device.binding.output.es2.geometry_generation == 32u);
  CHECK(device.binding.output.aes2.geometry_generation == 32u);
  CHECK(device.binding.workspace.es2_candidate.geometry_generation == 32u);
  CHECK(device.binding.workspace.aes2_candidate.geometry_generation == 32u);
  Downloaded after;
  CUDA_CHECK(download(host, device, after, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(after.plan_error != 0u);
  CHECK(before.pair_data == after.pair_data);
  CHECK(before.coordination == after.coordination);
  CHECK(before.overlap == after.overlap);
  CHECK(before.dipole == after.dipole);
  CHECK(before.quadrupole == after.quadrupole);
  CHECK(before.h0 == after.h0);
  CHECK(before.es2 == after.es2);
  CHECK(before.aes2 == after.aes2);
  CHECK(before.geometry_generations == after.geometry_generations);
  CHECK(before.operator_generations == after.operator_generations);

  Gfn2PreprocessingDeviceBinding alias = device.binding;
  alias.output.overlap = const_cast<double*>(alias.input.positions);
  CHECK(seal_gfn2_preprocessing_binding_cuda(alias).error ==
        Gfn2PreprocessingBindingError::kInvalidAlias);
  Gfn2PreprocessingDeviceBinding stale = device.binding;
  stale.workspace.h0_candidate = stale.workspace.overlap_candidate;
  CHECK(validate_gfn2_preprocessing_binding_cuda(stale).error ==
            Gfn2PreprocessingBindingError::kInvalidAlias ||
        validate_gfn2_preprocessing_binding_cuda(stale).error ==
            Gfn2PreprocessingBindingError::kStaleSeal);
  Gfn2PreprocessingDeviceBinding cross = device.binding;
  cross.plan.aes2.plan_token ^= 1u;
  CHECK(seal_gfn2_preprocessing_binding_cuda(cross).error ==
        Gfn2PreprocessingBindingError::kCrossPlan);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

struct GraphResources {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  ~GraphResources() {
    if (executable != nullptr) (void)cudaGraphExecDestroy(executable);
    if (graph != nullptr) (void)cudaGraphDestroy(graph);
  }
};

int test_graph_replay_changed_positions_stable_binding() {
  HostCase host;
  std::string error;
  CHECK(host.create(8, error));
  const std::vector<double> first = changed_positions(host, 0.004);
  const std::vector<double> second = changed_positions(host, 0.019);
  Reference first_expected;
  Reference second_expected;
  CHECK(host.evaluate(first, 41u, first_expected, error));
  CHECK(host.evaluate(second, 41u, second_expected, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const auto positions_address = device.binding.input.positions;
  const auto overlap_address = device.binding.output.overlap;
  const auto candidate_address = device.binding.workspace.overlap_candidate;
  const std::uint64_t seal = device.binding.binding_seal;

  GraphResources graph;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  const auto diagnostic = compose_gfn2_preprocessing_cuda(device.binding, 41u, stream);
  const cudaError_t capture_status = cudaStreamEndCapture(stream, &graph.graph);
  CHECK(diagnostic.success());
  CUDA_CHECK(capture_status);
  CUDA_CHECK(cudaGraphInstantiate(&graph.executable, graph.graph, 0));
  CHECK(validate_gfn2_preprocessing_binding_cuda(device.binding).success());

  CUDA_CHECK(device.upload_positions(first, stream));
  CUDA_CHECK(cudaGraphLaunch(graph.executable, stream));
  Downloaded actual;
  CUDA_CHECK(download(host, device, actual, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(matches_reference(actual, first_expected, 41u));

  CUDA_CHECK(device.upload_positions(second, stream));
  CUDA_CHECK(cudaGraphLaunch(graph.executable, stream));
  CUDA_CHECK(download(host, device, actual, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(matches_reference(actual, second_expected, 41u));
  CHECK(device.binding.input.positions == positions_address);
  CHECK(device.binding.output.overlap == overlap_address);
  CHECK(device.binding.workspace.overlap_candidate == candidate_address);
  CHECK(device.binding.binding_seal == seal);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status == cudaErrorNoDevice || count_status == cudaErrorInsufficientDriver ||
      device_count == 0) {
    (void)cudaGetLastError();
    return 0;
  }
  CUDA_CHECK(count_status);
  CUDA_CHECK(cudaSetDevice(0));
  for (const auto test : {test_cpu_parity_batches, test_peer_transaction_and_inactive_mask,
                          test_plan_failure_and_seal_fail_closed,
                          test_graph_replay_changed_positions_stable_binding}) {
    const int status = test();
    if (status != 0) return status;
  }
  return 0;
}
