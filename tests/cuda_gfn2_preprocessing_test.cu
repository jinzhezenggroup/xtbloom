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

using namespace xtbloom::detail;
using namespace xtbloom::detail::cuda;
using namespace xtbloom::detail::gfn2;

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
                               coordination_plan, error) != XTBLOOM_STATUS_SUCCESS ||
        make_basis_plan(batch, atoms, atom_offsets.data(), atomic_numbers.data(), basis, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        make_integral_plan(basis, integrals, error) != XTBLOOM_STATUS_SUCCESS ||
        make_h0_plan(basis, integrals, atomic_numbers.data(), h0_plan, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        make_es2_plan(basis, atomic_numbers.data(), es2_plan, error) != XTBLOOM_STATUS_SUCCESS ||
        make_aes2_plan(basis, atomic_numbers.data(), aes2_plan, error) != XTBLOOM_STATUS_SUCCESS) {
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
                                  error) != XTBLOOM_STATUS_SUCCESS) {
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
                             error) != XTBLOOM_STATUS_SUCCESS ||
        evaluate_multipole_cpu(basis, integrals, coordinates.data(), result.dipole.data(),
                               result.quadrupole.data(), integral_workspace.data(),
                               integral_workspace.size() * sizeof(double),
                               error) != XTBLOOM_STATUS_SUCCESS ||
        evaluate_h0_cpu(basis, integrals, h0_plan, coordinates.data(), result.coordination.data(),
                        result.overlap.data(), result.h0.data(), error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    std::vector<double> es2_scratch(result.es2.size());
    ES2Workspace es2_workspace{};
    es2_workspace.matrix_scratch = es2_scratch.data();
    es2_workspace.matrix_elements = es2_plan.total_matrix_elements();
    ES2GeometryCache es2_cache{};
    if (update_es2_geometry_cache_cpu(es2_plan, coordinates.data(), generation, result.es2.data(),
                                      result.es2.size(), es2_workspace, es2_cache,
                                      error) != XTBLOOM_STATUS_SUCCESS) {
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
                                          error) == XTBLOOM_STATUS_SUCCESS;
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
  DeviceBuffer<std::uint64_t> public_geometry_generation, public_operator_generation,
      geometry_epoch;

  DeviceBuffer<double> position_scratch, candidate_geometry_pair, candidate_coordination,
      geometry_pair_scratch, geometry_coordination_scratch, candidate_overlap, candidate_dipole,
      candidate_quadrupole, candidate_h0, integral_overlap_scratch, integral_dipole_scratch,
      integral_quadrupole_scratch, integral_h0_scratch, candidate_es2, es2_scratch, candidate_aes2,
      aes2_scratch;
  DeviceBuffer<std::uint64_t> candidate_geometry_generation;
  DeviceBuffer<std::uint32_t> geometry_sequence, integral_sequence, geometry_system_errors,
      geometry_device_error, integral_system_errors, integral_device_error, es2_device_error,
      aes2_system_errors, aes2_device_error, system_stages, plan_error;
  /* Sparse pair-list gate buffers, allocated only by enable_pairlist(). */
  DeviceBuffer<xtbloom::detail::Gfn2AtomPair> pairlist_pairs;
  DeviceBuffer<std::int64_t> pairlist_offsets, pairlist_neighbor_offsets, pairlist_neighbors,
      pairlist_atom_cells, pairlist_cell_counts, pairlist_cell_offsets, pairlist_cell_fill,
      pairlist_cell_atoms, pairlist_neighbor_cursor, pairlist_neighbor_scratch,
      pairlist_pair_cursor;
  DeviceBuffer<std::int64_t> pairlist_pair_counts, pairlist_neighbor_counts;
  DeviceBuffer<std::int32_t> pairlist_system_modes;
  DeviceBuffer<std::uint64_t> pairlist_generations;
  DeviceBuffer<xtbloom::detail::cuda::Gfn2PairListSystemMeta> pairlist_meta;
  DeviceBuffer<std::uint32_t> pairlist_sequence, sparse_system_errors, sparse_device_error;
  DeviceBuffer<double> sparse_coordination;
  /* Committed output pair-list storage (step 4), allocated by enable_pairlist(). */
  DeviceBuffer<xtbloom::detail::Gfn2AtomPair> committed_pairs;
  DeviceBuffer<std::int64_t> committed_pair_offsets, committed_pair_counts,
      committed_neighbor_offsets, committed_neighbor_counts, committed_neighbors;
  DeviceBuffer<std::uint64_t> committed_pair_generations;
  DeviceBuffer<std::uint8_t> committed_eligible_mask;

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
    ALLOC(geometry_epoch, 1u);
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
                           nullptr,
                           0,
                           nullptr,
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

  /* Wire the optional sparse pair-list leaf and its gate buffers.  Capacities
   * are conservative fixed-topology upper bounds: neighbors and pairs use the
   * all-pairs maximum and cells scale with the largest system, so the
   * documented overflow detection cannot trigger for a valid system.  After
   * this call the binding must be re-sealed before composing. */
  cudaError_t enable_pairlist(const HostCase& host, cudaStream_t stream) {
    const std::int64_t batch = host.batch;
    const std::int64_t atoms = host.basis.total_atoms;
    std::int64_t max_atoms = 0;
    for (std::int64_t system = 0; system < batch; ++system) {
      max_atoms = std::max(max_atoms, host.atom_offsets[static_cast<std::size_t>(system + 1)] -
                                          host.atom_offsets[static_cast<std::size_t>(system)]);
    }
    const std::int64_t max_pairs_per_system = max_atoms * (max_atoms - 1) / 2;
    const std::int64_t max_cells_per_system = std::max<std::int64_t>(16, 8 * max_atoms);
    const std::int64_t max_neighbors_per_atom = max_atoms;
    cudaError_t status = cudaSuccess;
#define PL_ALLOC(field, count) \
  if (status == cudaSuccess) status = field.allocate(static_cast<std::size_t>(count))
    PL_ALLOC(sparse_system_errors, batch);
    PL_ALLOC(sparse_device_error, 1u);
    PL_ALLOC(sparse_coordination, atoms);
    PL_ALLOC(pairlist_pairs, batch * max_pairs_per_system);
    PL_ALLOC(pairlist_offsets, batch + 1);
    PL_ALLOC(pairlist_pair_counts, batch);
    PL_ALLOC(pairlist_neighbor_offsets, atoms + 1);
    PL_ALLOC(pairlist_neighbor_counts, atoms);
    PL_ALLOC(pairlist_neighbors, atoms * max_neighbors_per_atom);
    PL_ALLOC(pairlist_generations, batch);
    PL_ALLOC(pairlist_meta, batch);
    PL_ALLOC(pairlist_atom_cells, atoms);
    PL_ALLOC(pairlist_system_modes, batch);
    const std::int64_t cell_storage = batch * (max_cells_per_system + 1);
    PL_ALLOC(pairlist_cell_counts, cell_storage);
    PL_ALLOC(pairlist_cell_offsets, cell_storage);
    PL_ALLOC(pairlist_cell_fill, cell_storage);
    PL_ALLOC(pairlist_cell_atoms, atoms);
    PL_ALLOC(pairlist_neighbor_cursor, atoms);
    PL_ALLOC(pairlist_neighbor_scratch, atoms * max_neighbors_per_atom);
    PL_ALLOC(pairlist_pair_cursor, batch);
    PL_ALLOC(pairlist_sequence, 1u);
    PL_ALLOC(committed_pairs, batch * max_pairs_per_system);
    PL_ALLOC(committed_pair_offsets, batch + 1);
    PL_ALLOC(committed_pair_counts, batch);
    PL_ALLOC(committed_neighbor_offsets, atoms + 1);
    PL_ALLOC(committed_neighbor_counts, atoms);
    PL_ALLOC(committed_neighbors, atoms * max_neighbors_per_atom);
    PL_ALLOC(committed_pair_generations, batch);
    PL_ALLOC(committed_eligible_mask, batch);
#undef PL_ALLOC
    if (status != cudaSuccess) return status;
    /* Committed payload capacity is caller-owned and only live prefixes are
     * published.  Seed the unused tails so full-capacity downloads can prove
     * transactional byte preservation under initcheck as well as CTest. */
    status = cudaMemsetAsync(committed_pairs.get(), 0xa5,
                             committed_pairs.size() * sizeof(Gfn2AtomPair), stream);
    if (status != cudaSuccess) return status;
    status = cudaMemsetAsync(committed_neighbors.get(), 0xa5,
                             committed_neighbors.size() * sizeof(std::int64_t), stream);
    if (status != cudaSuccess) return status;
    /* Setup initializes never-published metadata once. Repeated calls retain
     * the last committed counts/generation for failed or inactive peers. */
    status = committed_pair_counts.fill(0, stream);
    if (status != cudaSuccess) return status;
    status = committed_neighbor_counts.fill(0, stream);
    if (status != cudaSuccess) return status;
    status = committed_pair_generations.fill(0u, stream);
    if (status != cudaSuccess) return status;
    status = committed_eligible_mask.fill(0u, stream);
    if (status != cudaSuccess) return status;
    /* Host-set per-system dispatch decisions.  This fixture builds every system
     * dense (kDense) so the bucketed and all-pairs paths can be exercised and
     * compared; the binding requires system_modes when the leaf is enabled. */
    {
      std::vector<std::int32_t> modes(
          static_cast<std::size_t>(batch),
          static_cast<std::int32_t>(xtbloom::detail::cuda::Gfn2PairListMode::kDense));
      status = pairlist_system_modes.upload(modes.data(), modes.size(), stream);
      if (status != cudaSuccess) return status;
    }
    binding.plan.pairlist = {batch,
                             atoms,
                             batch + 1,
                             xtbloom::detail::cuda::kDefaultPairlistCutoffBohr,
                             max_cells_per_system,
                             max_neighbors_per_atom,
                             max_pairs_per_system,
                             xtbloom::detail::cuda::Gfn2PairListMode::kSparse,
                             kPlanToken,
                             atom_offsets.get(),
                             kGfn2PairListAllowDenseFallback,
                             pairlist_system_modes.get(),
                             batch};
    binding.diagnostics.sparse_system_errors = sparse_system_errors.get();
    binding.diagnostics.sparse_system_elements = batch;
    binding.diagnostics.sparse_device_error = sparse_device_error.get();
    binding.workspace.sparse_coordination = sparse_coordination.get();
    binding.workspace.sparse_coordination_elements = atoms;
    binding.workspace.pairlist_candidate = {pairlist_pairs.get(),
                                            batch * max_pairs_per_system,
                                            pairlist_offsets.get(),
                                            batch + 1,
                                            pairlist_pair_counts.get(),
                                            batch,
                                            pairlist_neighbor_offsets.get(),
                                            atoms + 1,
                                            pairlist_neighbor_counts.get(),
                                            atoms,
                                            pairlist_neighbors.get(),
                                            atoms * max_neighbors_per_atom,
                                            pairlist_generations.get(),
                                            batch,
                                            kPlanToken};
    binding.workspace.pairlist = {pairlist_meta.get(),
                                  batch,
                                  pairlist_atom_cells.get(),
                                  atoms,
                                  pairlist_cell_counts.get(),
                                  cell_storage,
                                  pairlist_cell_offsets.get(),
                                  cell_storage,
                                  pairlist_cell_fill.get(),
                                  cell_storage,
                                  pairlist_cell_atoms.get(),
                                  atoms,
                                  pairlist_neighbor_cursor.get(),
                                  atoms,
                                  pairlist_neighbor_scratch.get(),
                                  atoms * max_neighbors_per_atom,
                                  pairlist_pair_cursor.get(),
                                  batch,
                                  pairlist_sequence.get(),
                                  1,
                                  kPlanToken};
    binding.output.pairlist = {xtbloom::detail::Gfn2PlanMemorySpace::kCudaDevice,
                               xtbloom::detail::Gfn2PairListState::kCommitted,
                               xtbloom::detail::Gfn2PairListRole::kCoordination,
                               xtbloom::detail::Gfn2PairMapKind::kExplicit,
                               kPlanToken,
                               xtbloom::detail::cuda::kDefaultPairlistCutoffBohr,
                               xtbloom::detail::cuda::kDefaultPairlistCutoffBohr,
                               batch,
                               atoms,
                               max_pairs_per_system,
                               max_neighbors_per_atom,
                               batch + 1,
                               atoms + 1,
                               batch * max_pairs_per_system,
                               atoms * max_neighbors_per_atom,
                               committed_pair_offsets.get(),
                               committed_pairs.get(),
                               batch,
                               atoms,
                               committed_pair_counts.get(),
                               committed_neighbor_counts.get(),
                               committed_neighbor_offsets.get(),
                               committed_neighbors.get(),
                               batch,
                               batch,
                               0,
                               committed_pair_generations.get(),
                               committed_eligible_mask.get(),
                               nullptr};
    return cudaSuccess;
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
  std::vector<xtbloom::detail::Gfn2AtomPair> committed_pairs;
  std::vector<std::int64_t> committed_pair_offsets, committed_pair_counts,
      committed_neighbor_offsets, committed_neighbor_counts, committed_neighbors;
  std::vector<std::uint64_t> committed_generations;
  std::vector<std::uint8_t> committed_eligible;
  std::uint32_t plan_error = 0u;
};

cudaError_t download_committed(const HostCase& host, const DeviceFixture& device,
                               Downloaded& values, cudaStream_t stream) {
  values.committed_pairs.resize(device.committed_pairs.size());
  values.committed_pair_offsets.resize(device.committed_pair_offsets.size());
  values.committed_pair_counts.resize(device.committed_pair_counts.size());
  values.committed_neighbor_offsets.resize(device.committed_neighbor_offsets.size());
  values.committed_neighbor_counts.resize(device.committed_neighbor_counts.size());
  values.committed_neighbors.resize(device.committed_neighbors.size());
  values.committed_generations.resize(device.committed_pair_generations.size());
  values.committed_eligible.resize(device.committed_eligible_mask.size());
  cudaError_t status = device.committed_pairs.download(values.committed_pairs.data(),
                                                       values.committed_pairs.size(), stream);
#define DOWNLOAD_C(field, target) \
  if (status == cudaSuccess)      \
  status = device.field.download(values.target.data(), values.target.size(), stream)
  DOWNLOAD_C(committed_pair_offsets, committed_pair_offsets);
  DOWNLOAD_C(committed_pair_counts, committed_pair_counts);
  DOWNLOAD_C(committed_neighbor_offsets, committed_neighbor_offsets);
  DOWNLOAD_C(committed_neighbor_counts, committed_neighbor_counts);
  DOWNLOAD_C(committed_neighbors, committed_neighbors);
  DOWNLOAD_C(committed_pair_generations, committed_generations);
  DOWNLOAD_C(committed_eligible_mask, committed_eligible);
#undef DOWNLOAD_C
  return status;
}

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

bool published_state_equal(const Downloaded& first, const Downloaded& second) {
  const auto same = [](const auto& left, const auto& right, const char* name) {
    if (left == right) return true;
    std::fprintf(stderr, "published state changed: %s\n", name);
    return false;
  };
#define CHECK_SAME(field) \
  if (!same(first.field, second.field, #field)) return false
  CHECK_SAME(pair_data);
  CHECK_SAME(coordination);
  CHECK_SAME(overlap);
  CHECK_SAME(dipole);
  CHECK_SAME(quadrupole);
  CHECK_SAME(h0);
  CHECK_SAME(es2);
  CHECK_SAME(aes2);
  CHECK_SAME(geometry_generations);
  CHECK_SAME(operator_generations);
  CHECK_SAME(committed_pair_offsets);
  CHECK_SAME(committed_pair_counts);
  CHECK_SAME(committed_neighbor_offsets);
  CHECK_SAME(committed_neighbor_counts);
  CHECK_SAME(committed_neighbors);
  CHECK_SAME(committed_generations);
  CHECK_SAME(committed_eligible);
#undef CHECK_SAME
  if (first.committed_pairs.size() != second.committed_pairs.size()) return false;
  for (std::size_t index = 0; index < first.committed_pairs.size(); ++index) {
    if (first.committed_pairs[index].first != second.committed_pairs[index].first ||
        first.committed_pairs[index].second != second.committed_pairs[index].second) {
      return false;
    }
  }
  return true;
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

bool stable_binding_except_attempted_generation(const Gfn2PreprocessingDeviceBinding& first,
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
  return std::memcmp(&normalized_first, &normalized_second, sizeof(normalized_first)) == 0;
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
  const std::uint64_t initial_epoch = 21u;
  CUDA_CHECK(device.geometry_epoch.upload(&initial_epoch, 1u, stream));
  device.binding.geometry_epoch = {device.geometry_epoch.get(), 1, kPlanToken};
  CHECK(seal_gfn2_preprocessing_binding_cuda(device.binding).success());
  CHECK(compose_gfn2_preprocessing_epoch_cuda(device.binding, stream).success());
  Downloaded after;
  std::uint64_t actual_epoch = 0u;
  CUDA_CHECK(download(host, device, after, stream));
  CUDA_CHECK(device.geometry_epoch.download(&actual_epoch, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(actual_epoch == 22u);

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
  CUDA_CHECK(device.enable_pairlist(host, stream));
  CHECK(seal_gfn2_preprocessing_binding_cuda(device.binding).success());

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
  Downloaded committed_31;
  CUDA_CHECK(download_committed(host, device, committed_31, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  /* The committed sparse view must follow the complete operator gate, not a
   * geometry-only health check. A finite maximal shell level passes parameter
   * preflight but overflows the H0 arithmetic for its owning peer. This keeps
   * the failure peer-local and proves pair-list eligibility mirrors the final
   * publication decision. */
  const std::int64_t failed_shell = host.basis.batch_shell_offsets[0];
  const double overflowing_level = std::numeric_limits<double>::max();
  CUDA_CHECK(cudaMemcpyAsync(device.shell_levels.get() + failed_shell, &overflowing_level,
                             sizeof(overflowing_level), cudaMemcpyHostToDevice, stream));
  constexpr std::int64_t kFailed = 0;
  constexpr std::int64_t kInactive = 1;
  std::vector<std::uint8_t> requested(static_cast<std::size_t>(host.batch), 1u);
  requested[static_cast<std::size_t>(kInactive)] = 0u;
  CUDA_CHECK(device.requested.upload(requested.data(), requested.size(), stream));
  CHECK(compose_gfn2_preprocessing_cuda(device.binding, 32u, stream).success());
  Downloaded operator_failed;
  CUDA_CHECK(download(host, device, operator_failed, stream));
  CUDA_CHECK(download_committed(host, device, operator_failed, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (std::int64_t system = 0; system < host.batch; ++system) {
    const std::size_t index = static_cast<std::size_t>(system);
    CHECK(operator_failed.committed_eligible[index] == operator_failed.published[index]);
    if (system == kFailed || system == kInactive) {
      CHECK(operator_failed.published[index] == 0u);
      CHECK(operator_failed.committed_generations[index] == 31u);
      CHECK(operator_failed.committed_pair_counts[index] ==
            committed_31.committed_pair_counts[index]);
      const std::int64_t atom_begin = host.atom_offsets[index];
      const std::int64_t atom_end = host.atom_offsets[index + 1u];
      for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
        CHECK(operator_failed.committed_neighbor_counts[static_cast<std::size_t>(atom)] ==
              committed_31.committed_neighbor_counts[static_cast<std::size_t>(atom)]);
      }
    } else {
      CHECK(operator_failed.published[index] == 1u);
      CHECK(operator_failed.committed_generations[index] == 32u);
    }
  }
  CHECK(operator_failed.committed_pairs.size() == committed_31.committed_pairs.size());
  for (std::size_t pair = 0; pair < operator_failed.committed_pairs.size(); ++pair) {
    CHECK(operator_failed.committed_pairs[pair].first == committed_31.committed_pairs[pair].first);
    CHECK(operator_failed.committed_pairs[pair].second ==
          committed_31.committed_pairs[pair].second);
  }
  CHECK(operator_failed.committed_neighbors == committed_31.committed_neighbors);
  const double valid_level = host.h0_plan.shell_levels[static_cast<std::size_t>(failed_shell)];
  CUDA_CHECK(cudaMemcpyAsync(device.shell_levels.get() + failed_shell, &valid_level,
                             sizeof(valid_level), cudaMemcpyHostToDevice, stream));
  std::fill(requested.begin(), requested.end(), 1u);
  CUDA_CHECK(device.requested.upload(requested.data(), requested.size(), stream));

  const std::int64_t malformed = host.basis.total_atoms + 1;
  CUDA_CHECK(cudaMemcpyAsync(device.atom_offsets.get() + host.batch, &malformed, sizeof(malformed),
                             cudaMemcpyHostToDevice, stream));
  CHECK(compose_gfn2_preprocessing_cuda(device.binding, 33u, stream).success());
  CHECK(device.binding.output.es2.geometry_generation == 33u);
  CHECK(device.binding.output.aes2.geometry_generation == 33u);
  CHECK(device.binding.workspace.es2_candidate.geometry_generation == 33u);
  CHECK(device.binding.workspace.aes2_candidate.geometry_generation == 33u);
  Downloaded after;
  CUDA_CHECK(download(host, device, after, stream));
  CUDA_CHECK(download_committed(host, device, after, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(after.plan_error != 0u);
  CHECK(operator_failed.pair_data == after.pair_data);
  CHECK(operator_failed.coordination == after.coordination);
  CHECK(operator_failed.overlap == after.overlap);
  CHECK(operator_failed.dipole == after.dipole);
  CHECK(operator_failed.quadrupole == after.quadrupole);
  CHECK(operator_failed.h0 == after.h0);
  CHECK(operator_failed.es2 == after.es2);
  CHECK(operator_failed.aes2 == after.aes2);
  CHECK(operator_failed.geometry_generations == after.geometry_generations);
  CHECK(operator_failed.operator_generations == after.operator_generations);
  CHECK(operator_failed.committed_pairs.size() == after.committed_pairs.size());
  for (std::size_t pair = 0; pair < operator_failed.committed_pairs.size(); ++pair) {
    CHECK(operator_failed.committed_pairs[pair].first == after.committed_pairs[pair].first);
    CHECK(operator_failed.committed_pairs[pair].second == after.committed_pairs[pair].second);
  }
  CHECK(operator_failed.committed_pair_offsets == after.committed_pair_offsets);
  CHECK(operator_failed.committed_pair_counts == after.committed_pair_counts);
  CHECK(operator_failed.committed_neighbor_offsets == after.committed_neighbor_offsets);
  CHECK(operator_failed.committed_neighbor_counts == after.committed_neighbor_counts);
  CHECK(operator_failed.committed_neighbors == after.committed_neighbors);
  CHECK(operator_failed.committed_generations == after.committed_generations);
  CHECK(operator_failed.committed_eligible == after.committed_eligible);

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
  Gfn2PreprocessingDeviceBinding epoch_alias = device.binding;
  epoch_alias.geometry_epoch = {device.public_operator_generation.get(), 1, kPlanToken};
  CHECK(seal_gfn2_preprocessing_binding_cuda(epoch_alias).error ==
        Gfn2PreprocessingBindingError::kInvalidAlias);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_sparse_plan_failure_preserves_publication() {
  HostCase host;
  std::string error;
  CHECK(host.create(8, error));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CUDA_CHECK(device.enable_pairlist(host, stream));
  CHECK(seal_gfn2_preprocessing_binding_cuda(device.binding).success());

  CHECK(compose_gfn2_preprocessing_cuda(device.binding, 71u, stream).success());
  Downloaded before;
  CUDA_CHECK(download(host, device, before, stream));
  CUDA_CHECK(download_committed(host, device, before, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(before.plan_error == 0u);

  /* system_modes is a shared dispatch descriptor validated by pair-list
   * preflight.  A hostile device value must classify as one plan-wide failure
   * before any public or committed metadata initializer can run. */
  std::vector<std::int32_t> modes(device.pairlist_system_modes.size());
  CUDA_CHECK(device.pairlist_system_modes.download(modes.data(), modes.size(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  modes[0] = std::numeric_limits<std::int32_t>::max();
  CUDA_CHECK(device.pairlist_system_modes.upload(modes.data(), modes.size(), stream));
  CHECK(compose_gfn2_preprocessing_cuda(device.binding, 72u, stream).success());

  Downloaded after;
  CUDA_CHECK(download(host, device, after, stream));
  CUDA_CHECK(download_committed(host, device, after, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(after.plan_error ==
        static_cast<std::uint32_t>(Gfn2PreprocessingDeviceError::kSparsePairlistFailure));
  CHECK(published_state_equal(before, after));
  CHECK(std::all_of(after.published.begin(), after.published.end(),
                    [](std::uint8_t value) { return value == 0u; }));

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
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    HostCase host;
    std::string error;
    CHECK(host.create(batch_size, error));
    const std::vector<double> first = changed_positions(host, 0.004);
    const std::vector<double> second = changed_positions(host, 0.019);
    Reference first_expected;
    Reference second_expected;
    CHECK(host.evaluate(first, 41u, first_expected, error));
    CHECK(host.evaluate(second, 42u, second_expected, error));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    DeviceFixture device;
    CUDA_CHECK(device.initialize(host, stream));
    const std::uint64_t initial_epoch = 40u;
    CUDA_CHECK(device.geometry_epoch.upload(&initial_epoch, 1u, stream));
    device.binding.geometry_epoch = {device.geometry_epoch.get(), 1, kPlanToken};
    CHECK(seal_gfn2_preprocessing_binding_cuda(device.binding).success());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto positions_address = device.binding.input.positions;
    const auto overlap_address = device.binding.output.overlap;
    const auto candidate_address = device.binding.workspace.overlap_candidate;
    const auto epoch_address = device.binding.geometry_epoch.value;
    const std::uint64_t seal = device.binding.binding_seal;

    GraphResources graph;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    const auto diagnostic = compose_gfn2_preprocessing_epoch_cuda(device.binding, stream);
    const cudaError_t capture_status = cudaStreamEndCapture(stream, &graph.graph);
    CHECK(diagnostic.success());
    CUDA_CHECK(capture_status);
    CUDA_CHECK(cudaGraphInstantiate(&graph.executable, graph.graph, 0));
    CHECK(validate_gfn2_preprocessing_binding_cuda(device.binding).success());

    CUDA_CHECK(device.upload_positions(first, stream));
    CUDA_CHECK(cudaGraphLaunch(graph.executable, stream));
    Downloaded actual;
    CUDA_CHECK(download(host, device, actual, stream));
    std::uint64_t actual_epoch = 0u;
    CUDA_CHECK(device.geometry_epoch.download(&actual_epoch, 1u, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(actual_epoch == 41u);
    CHECK(matches_reference(actual, first_expected, 41u));

    CUDA_CHECK(device.upload_positions(second, stream));
    CUDA_CHECK(cudaGraphLaunch(graph.executable, stream));
    CUDA_CHECK(download(host, device, actual, stream));
    CUDA_CHECK(device.geometry_epoch.download(&actual_epoch, 1u, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(actual_epoch == 42u);
    CHECK(matches_reference(actual, second_expected, 42u));

    const Downloaded before_overflow = actual;
    const std::uint64_t maximum_epoch = std::numeric_limits<std::uint64_t>::max();
    CUDA_CHECK(device.geometry_epoch.upload(&maximum_epoch, 1u, stream));
    CUDA_CHECK(cudaGraphLaunch(graph.executable, stream));
    CUDA_CHECK(download(host, device, actual, stream));
    CUDA_CHECK(device.geometry_epoch.download(&actual_epoch, 1u, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(actual_epoch == maximum_epoch);
    CHECK(actual.plan_error ==
          static_cast<std::uint32_t>(Gfn2PreprocessingDeviceError::kGeometryEpochOverflow));
    CHECK(std::all_of(actual.published.begin(), actual.published.end(),
                      [](std::uint8_t value) { return value == 0u; }));
    CHECK(before_overflow.pair_data == actual.pair_data);
    CHECK(before_overflow.coordination == actual.coordination);
    CHECK(before_overflow.overlap == actual.overlap);
    CHECK(before_overflow.dipole == actual.dipole);
    CHECK(before_overflow.quadrupole == actual.quadrupole);
    CHECK(before_overflow.h0 == actual.h0);
    CHECK(before_overflow.es2 == actual.es2);
    CHECK(before_overflow.aes2 == actual.aes2);
    CHECK(before_overflow.geometry_generations == actual.geometry_generations);
    CHECK(before_overflow.operator_generations == actual.operator_generations);
    CHECK(device.binding.input.positions == positions_address);
    CHECK(device.binding.output.overlap == overlap_address);
    CHECK(device.binding.workspace.overlap_candidate == candidate_address);
    CHECK(device.binding.geometry_epoch.value == epoch_address);
    CHECK(device.binding.binding_seal == seal);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
  return 0;
}

/*
 * The optional sparse pair-list consistency gate.  A correctly wired enabled
 * binding validates, seals, and composes with the gate running -- every
 * healthy peer's sparse coordination numbers agree bitwise with the dense
 * geometry cache, so the peer publishes normally.  A clobbered sparse
 * output must instead fail that peer closed through the geometry error slot,
 * which the enclosing publication gate turns into a skipped commit.
 */
int test_sparse_pairlist_gate() {
  for (const std::int64_t batch : {4}) {
    HostCase host;
    std::string error;
    CHECK(host.create(batch, error));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    DeviceFixture device;
    CUDA_CHECK(device.initialize(host, stream));
    CHECK(seal_gfn2_preprocessing_binding_cuda(device.binding).success());
    /* A disabled sparse leaf is the exact default state.  Any hidden pointer
     * is stale after sealing and cannot be accepted by resealing either. */
    Gfn2PreprocessingDeviceBinding stale_disabled_candidate = device.binding;
    stale_disabled_candidate.workspace.pairlist_candidate.pair_offsets = device.atom_offsets.get();
    CHECK(validate_gfn2_preprocessing_binding_cuda(stale_disabled_candidate).error ==
          Gfn2PreprocessingBindingError::kStaleSeal);
    stale_disabled_candidate.binding_seal = 0u;
    CHECK(seal_gfn2_preprocessing_binding_cuda(stale_disabled_candidate).error ==
          Gfn2PreprocessingBindingError::kCrossPlan);
    Gfn2PreprocessingDeviceBinding stale_disabled_output = device.binding;
    stale_disabled_output.output.pairlist.neighbor_counts = device.atom_offsets.get();
    CHECK(validate_gfn2_preprocessing_binding_cuda(stale_disabled_output).error ==
          Gfn2PreprocessingBindingError::kStaleSeal);
    stale_disabled_output.binding_seal = 0u;
    CHECK(seal_gfn2_preprocessing_binding_cuda(stale_disabled_output).error ==
          Gfn2PreprocessingBindingError::kCrossPlan);
    const Gfn2PreprocessingLaunchDiagnostic disabled_gate_diagnostic =
        gate_gfn2_sparse_coordination_cuda(device.binding, stream);
    CHECK(!disabled_gate_diagnostic.success());
    CHECK(disabled_gate_diagnostic.binding.error ==
          Gfn2PreprocessingBindingError::kInvalidWorkspace);
    CHECK(disabled_gate_diagnostic.binding.field == Gfn2PreprocessingBindingField::kPairlist);
    CHECK(disabled_gate_diagnostic.cuda_status == cudaErrorInvalidValue);
    CUDA_CHECK(device.enable_pairlist(host, stream));
    CHECK(seal_gfn2_preprocessing_binding_cuda(device.binding).success());
    /* Pair-generation storage is a stable binding field, even though the
     * device values change per refresh.  Changing its extent after sealing
     * must invalidate the seal rather than silently changing the graph view. */
    Gfn2PreprocessingDeviceBinding stale_generation = device.binding;
    stale_generation.workspace.pairlist_candidate.generation_elements += 1;
    CHECK(validate_gfn2_preprocessing_binding_cuda(stale_generation).error ==
          Gfn2PreprocessingBindingError::kStaleSeal);
    /* A short cell-atom arena must fail during descriptor validation, before
     * any preprocessing kernel is queued. */
    Gfn2PreprocessingDeviceBinding short_cell_atoms = device.binding;
    short_cell_atoms.workspace.pairlist.cell_atom_elements = host.basis.total_atoms - 1;
    short_cell_atoms.binding_seal = 0u;
    CHECK(seal_gfn2_preprocessing_binding_cuda(short_cell_atoms).error ==
          Gfn2PreprocessingBindingError::kInvalidWorkspace);
    CHECK(compose_gfn2_preprocessing_cuda(device.binding, 61u, stream).success());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    Downloaded first;
    CUDA_CHECK(download(host, device, first, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(first.plan_error == 0u);
    for (std::int64_t system = 0; system < batch; ++system) {
      CHECK(first.geometry_errors[static_cast<std::size_t>(system)] ==
            static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess));
      CHECK(first.published[static_cast<std::size_t>(system)] == 1u);
    }

    /* Step 4 committed pair-list transaction: after the per-system gate, the
     * sparse candidate list is published into the stable output consumer view
     * with per-peer eligibility and the current generation.  All requested
     * healthy peers are eligible.  Offsets are fixed-capacity slot starts and
     * explicit counts delimit each copied candidate prefix. */
    CHECK(seal_gfn2_preprocessing_binding_cuda(device.binding).success());
    Downloaded committed_view;
    CUDA_CHECK(download_committed(host, device, committed_view, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::int64_t system = 0; system < batch; ++system) {
      CHECK(committed_view.committed_eligible[static_cast<std::size_t>(system)] == 1u);
      CHECK(committed_view.committed_generations[static_cast<std::size_t>(system)] == 61u);
    }
    const std::int64_t pair_stride = device.binding.plan.pairlist.max_pairs_per_system;
    const std::int64_t neighbor_stride = device.binding.plan.pairlist.max_neighbors_per_atom;
    for (std::int64_t system = 0; system < batch; ++system) {
      CHECK(committed_view.committed_pair_offsets[static_cast<std::size_t>(system)] ==
            system * pair_stride);
      const std::int64_t count =
          committed_view.committed_pair_counts[static_cast<std::size_t>(system)];
      CHECK(count >= 0);
      const std::int64_t atom_begin = host.atom_offsets[static_cast<std::size_t>(system)];
      const std::int64_t atom_end = host.atom_offsets[static_cast<std::size_t>(system + 1)];
      for (std::int64_t index = 0; index < count; ++index) {
        const auto pair =
            committed_view.committed_pairs[static_cast<std::size_t>(system * pair_stride + index)];
        CHECK(pair.first >= atom_begin && pair.first < pair.second && pair.second < atom_end);
      }
    }
    CHECK(committed_view.committed_pair_offsets[static_cast<std::size_t>(batch)] ==
          batch * pair_stride);
    for (std::int64_t atom_index = 0; atom_index < host.basis.total_atoms; ++atom_index) {
      CHECK(committed_view.committed_neighbor_offsets[static_cast<std::size_t>(atom_index)] ==
            atom_index * neighbor_stride);
      const std::int64_t count =
          committed_view.committed_neighbor_counts[static_cast<std::size_t>(atom_index)];
      for (std::int64_t index = 0; index < count; ++index) {
        const std::int64_t peer = committed_view.committed_neighbors[static_cast<std::size_t>(
            atom_index * neighbor_stride + index)];
        CHECK(peer >= 0 && peer < host.basis.total_atoms);
      }
    }
    CHECK(committed_view
              .committed_neighbor_offsets[static_cast<std::size_t>(host.basis.total_atoms)] ==
          host.basis.total_atoms * neighbor_stride);

    Gfn2PreprocessingDeviceBinding aliased = device.binding;
    aliased.binding_seal = 0u;
    aliased.workspace.pairlist_candidate.pairs = reinterpret_cast<xtbloom::detail::Gfn2AtomPair*>(
        const_cast<double*>(aliased.input.positions));
    const Gfn2PreprocessingBindingDiagnostic alias_diagnostic =
        seal_gfn2_preprocessing_binding_cuda(aliased);
    CHECK(alias_diagnostic.error == Gfn2PreprocessingBindingError::kInvalidAlias);
    CHECK(alias_diagnostic.field == Gfn2PreprocessingBindingField::kWorkspace);

    /* The standalone gate is an execution entry point too.  A sealed binding
     * whose pairlist scalar is mutated must be rejected before the gate kernel
     * is queued, just like the full composer. */
    Gfn2PreprocessingDeviceBinding stale_gate = device.binding;
    stale_gate.plan.pairlist.cutoff += 1.0;
    const Gfn2PreprocessingLaunchDiagnostic stale_gate_diagnostic =
        gate_gfn2_sparse_coordination_cuda(stale_gate, stream);
    CHECK(!stale_gate_diagnostic.success());
    CHECK(stale_gate_diagnostic.binding.error == Gfn2PreprocessingBindingError::kStaleSeal);
    CHECK(stale_gate_diagnostic.binding.field == Gfn2PreprocessingBindingField::kSeal);
    CHECK(stale_gate_diagnostic.cuda_status == cudaErrorInvalidValue);

    Gfn2PreprocessingDeviceBinding invalid_builder_cutoff = device.binding;
    invalid_builder_cutoff.binding_seal = 0u;
    invalid_builder_cutoff.plan.pairlist.cutoff = std::nextafter(kDefaultPairlistCutoffBohr, 0.0);
    CHECK(seal_gfn2_preprocessing_binding_cuda(invalid_builder_cutoff).error ==
          Gfn2PreprocessingBindingError::kInvalidWorkspace);
    Gfn2PreprocessingDeviceBinding mismatched_consumer_cutoff = device.binding;
    mismatched_consumer_cutoff.binding_seal = 0u;
    mismatched_consumer_cutoff.output.pairlist.list_builder_cutoff_bohr =
        std::nextafter(kDefaultPairlistCutoffBohr, 0.0);
    CHECK(seal_gfn2_preprocessing_binding_cuda(mismatched_consumer_cutoff).error ==
          Gfn2PreprocessingBindingError::kInvalidExtent);

    /* Run the gate in isolation with a deliberately clobbered sparse output
     * to prove the fail-closed flag: the composed dense geometry candidate is
     * untouched, the corrupted sparse coordination disagrees bitwise, and the
     * gate records kSparseCoordinationMismatch for exactly the corrupted peer.
     * Healthy peers keep a clean geometry-error slot.  (Rejection of any
     * nonzero geometry error by the publication pass is covered by the peer
     * transaction test; the standalone gate entry intentionally stops before
     * re-publishing.) */
    const double wrong = -123.5;
    CUDA_CHECK(device.sparse_coordination.upload(&wrong, 1u, stream));
    CHECK(gate_gfn2_sparse_coordination_cuda(device.binding, stream).success());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    Downloaded second;
    CUDA_CHECK(download(host, device, second, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(second.geometry_errors[0] ==
          static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSparseCoordinationMismatch));
    for (std::int64_t system = 1; system < batch; ++system) {
      CHECK(second.geometry_errors[static_cast<std::size_t>(system)] ==
            static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
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
  for (const auto test :
       {test_cpu_parity_batches, test_peer_transaction_and_inactive_mask,
        test_plan_failure_and_seal_fail_closed, test_sparse_plan_failure_preserves_publication,
        test_graph_replay_changed_positions_stable_binding, test_sparse_pairlist_gate}) {
    const int status = test();
    if (status != 0) return status;
  }
  return 0;
}
