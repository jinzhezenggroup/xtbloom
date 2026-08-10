#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_inference_publication.cuh"
#include "model/gfn2/coordination.hpp"
#include "model/gfn2/es3.hpp"
#include "model/gfn2/force.hpp"
#include "model/gfn2/repulsion.hpp"
#include "runtime/gfn2_cuda_execution.hpp"
#include "tests/support/gfn2_scc_test_case.hpp"
#include "xtbloom/xtbloom.h"

namespace {

using xtbloom::detail::Gfn2CudaExecutionCache;
using xtbloom::detail::Gfn2CudaExecutionIdentity;
using xtbloom::detail::Gfn2CudaSccStartMode;
using xtbloom::detail::cuda::Gfn2InferencePublicationPlanError;
using xtbloom::detail::cuda::Gfn2InferencePublicationSystemError;
using xtbloom::test::gfn2::HostSccCase;
using xtbloom::test::gfn2::HostSccCaseOptions;
using xtbloom::test::gfn2::SmallSystemKind;
namespace gfn2 = xtbloom::detail::gfn2;

constexpr std::array<std::int64_t, 4> kBatchSizes{1, 8, 32, 128};
constexpr std::uint32_t kAllResultFlags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES |
                                          XTBLOOM_COMPUTE_ATOMIC_CHARGES |
                                          XTBLOOM_COMPUTE_POINT_CHARGE_FORCES;

/*
 * These bounds cover different reduction/eigensolver orders while remaining
 * much smaller than the requested SCC convergence threshold. Energy is in
 * Hartree, forces in Hartree/bohr, and charges in elementary-charge units.
 */
constexpr double kEnergyAbsoluteTolerance = 3.0e-8;
constexpr double kEnergyRelativeTolerance = 3.0e-8;
constexpr double kChargeAbsoluteTolerance = 1.0e-7;
constexpr double kChargeRelativeTolerance = 1.0e-7;
constexpr double kForceAbsoluteTolerance = 3.0e-7;
constexpr double kForceRelativeTolerance = 3.0e-7;

const char* g_scenario = "uninitialized";

#define CHECK(condition)                                                                   \
  do {                                                                                     \
    if (!(condition)) {                                                                    \
      std::fprintf(stderr, "runtime parity check failed in %s at %s:%d: %s\n", g_scenario, \
                   __FILE__, __LINE__, #condition);                                        \
      return __LINE__;                                                                     \
    }                                                                                      \
  } while (false)

#define CUDA_CHECK(expression)                                                             \
  do {                                                                                     \
    const cudaError_t cuda_status_ = (expression);                                         \
    if (cuda_status_ != cudaSuccess) {                                                     \
      std::fprintf(stderr, "runtime parity CUDA failure in %s at %s:%d: %s\n", g_scenario, \
                   __FILE__, __LINE__, cudaGetErrorString(cuda_status_));                  \
      return __LINE__;                                                                     \
    }                                                                                      \
  } while (false)

template <typename T>
xtbloom_const_buffer_t input_buffer(const std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), XTBLOOM_MEMORY_HOST,
          0u};
}

template <typename T>
xtbloom_buffer_t output_buffer(std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), XTBLOOM_MEMORY_HOST,
          0u};
}

struct CpuContextDeleter {
  void operator()(xtbloom_context_t* context) const noexcept { xtbloom_context_destroy(context); }
};

using CpuContext = std::unique_ptr<xtbloom_context_t, CpuContextDeleter>;

/* Host model workspaces require at least double alignment; 64 bytes also
 * matches the fixture and eigensolver cache contracts. */
class AlignedBuffer {
 public:
  AlignedBuffer() noexcept = default;
  ~AlignedBuffer() { std::free(data_); }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  bool allocate(std::size_t requested) noexcept {
    if (data_ != nullptr || requested > std::numeric_limits<std::size_t>::max() - 63u) {
      return false;
    }
    bytes_ = (std::max<std::size_t>(requested, 1u) + 63u) & ~std::size_t{63u};
    data_ = std::aligned_alloc(64u, bytes_);
    if (data_ == nullptr) {
      bytes_ = 0u;
      return false;
    }
    std::memset(data_, 0, bytes_);
    return true;
  }

  void* data() noexcept { return data_; }
  std::size_t size() const noexcept { return bytes_; }

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0u;
};

CpuContext make_cpu_context() {
  xtbloom_context_options_t options{};
  if (xtbloom_context_options_init(&options, sizeof(options)) != XTBLOOM_STATUS_SUCCESS) {
    return {};
  }
  options.backend = XTBLOOM_BACKEND_CPU;
  xtbloom_context_t* raw = nullptr;
  if (xtbloom_context_create(&options, &raw) != XTBLOOM_STATUS_SUCCESS) {
    return {};
  }
  return CpuContext(raw);
}

xtbloom_compute_options_t compute_options() noexcept {
  xtbloom_compute_options_t options{};
  (void)xtbloom_compute_options_init(&options, sizeof(options));
  options.model = XTBLOOM_MODEL_GFN2_XTB;
  options.flags = kAllResultFlags;
  options.max_scc_iterations = 64;
  options.charge_tolerance = 1.0e-8;
  options.energy_tolerance = 1.0e-8;
  options.electronic_temperature = 0.0;
  return options;
}

/* Own every host input so descriptor addresses remain stable across setup,
 * CPU reference execution, asynchronous CUDA refresh, and result download. */
struct PublicHostBatch {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int64_t> point_offsets;
  std::vector<double> point_positions;
  std::vector<double> point_values;
  std::vector<double> point_gammas;
  std::vector<double> periodic_shifts;
  std::vector<std::int64_t> response_offsets;
  std::vector<double> response_matrix;
  xtbloom_batch_t descriptor{};

  void bind() noexcept {
    (void)xtbloom_batch_init(&descriptor, sizeof(descriptor));
    descriptor.batch_size = static_cast<std::int64_t>(molecular_charges.size());
    descriptor.total_atoms = static_cast<std::int64_t>(atomic_numbers.size());
    descriptor.total_point_charges = static_cast<std::int64_t>(point_values.size());
    descriptor.total_charge_response_elements = static_cast<std::int64_t>(response_matrix.size());
    descriptor.atom_offsets = input_buffer(atom_offsets);
    descriptor.atomic_numbers = input_buffer(atomic_numbers);
    descriptor.positions = input_buffer(positions);
    descriptor.molecular_charges = input_buffer(molecular_charges);
    descriptor.unpaired_electrons = input_buffer(unpaired_electrons);
    descriptor.point_charge_offsets = input_buffer(point_offsets);
    descriptor.point_charge_positions = input_buffer(point_positions);
    descriptor.point_charge_values = input_buffer(point_values);
    descriptor.point_charge_gammas = input_buffer(point_gammas);
    descriptor.atomic_potential_shifts = input_buffer(periodic_shifts);
    descriptor.charge_response_offsets = input_buffer(response_offsets);
    descriptor.charge_response_matrix = input_buffer(response_matrix);
  }

  static PublicHostBatch from_fixture(const HostSccCase& host, bool periodic) {
    PublicHostBatch batch;
    batch.atom_offsets = host.atom_offsets();
    batch.atomic_numbers = host.atomic_numbers();
    batch.positions = host.positions();
    batch.molecular_charges = host.molecular_charges();
    batch.unpaired_electrons = host.unpaired_electrons();
    batch.point_offsets = host.point_charge_offsets();
    batch.point_positions = host.point_charge_positions();
    batch.point_values = host.point_charge_charges();
    batch.point_gammas = host.point_charge_hardnesses();
    if (periodic) {
      batch.periodic_shifts = host.periodic_shifts();
      batch.response_matrix = host.periodic_response_matrices();
      batch.response_offsets.assign(static_cast<std::size_t>(host.batch_size() + 1), 0);
      for (std::int64_t system = 0; system < host.batch_size(); ++system) {
        const std::int64_t atoms = host.atom_offsets()[static_cast<std::size_t>(system + 1)] -
                                   host.atom_offsets()[static_cast<std::size_t>(system)];
        batch.response_offsets[static_cast<std::size_t>(system + 1)] =
            batch.response_offsets[static_cast<std::size_t>(system)] + atoms * atoms;
      }
    }
    batch.bind();
    return batch;
  }

  /* Change each ragged member without changing any topology-defining bytes. */
  void perturb_geometry() noexcept {
    const std::int64_t batch = static_cast<std::int64_t>(atom_offsets.size() - 1u);
    for (std::int64_t system = 0; system < batch; ++system) {
      const std::int64_t atom_begin = atom_offsets[static_cast<std::size_t>(system)];
      const std::int64_t atom_end = atom_offsets[static_cast<std::size_t>(system + 1)];
      const double step = 0.006 + 0.001 * static_cast<double>(system % 4);
      positions[3u * static_cast<std::size_t>(atom_begin)] -= step;
      if (atom_end - atom_begin > 1) {
        positions[3u * static_cast<std::size_t>(atom_end - 1)] += 0.75 * step;
      } else {
        positions[3u * static_cast<std::size_t>(atom_begin) + 1u] += 0.5 * step;
      }
    }
    bind();
  }
};

struct ReferenceResult {
  std::vector<double> energies;
  std::vector<double> qm_forces;
  std::vector<double> atomic_charges;
  std::vector<double> point_forces;
  std::vector<std::int32_t> iterations;
  std::vector<std::uint8_t> converged;
  std::vector<std::int32_t> statuses;
  xtbloom_batch_result_t descriptor{};

  void bind(const PublicHostBatch& batch) {
    const std::size_t systems = static_cast<std::size_t>(batch.descriptor.batch_size);
    energies.assign(systems, 0.0);
    qm_forces.assign(3u * batch.atomic_numbers.size(), 0.0);
    atomic_charges.assign(batch.atomic_numbers.size(), 0.0);
    point_forces.assign(3u * batch.point_values.size(), 0.0);
    iterations.assign(systems, 0);
    converged.assign(systems, 0u);
    statuses.assign(systems, XTBLOOM_STATUS_INTERNAL_ERROR);
    (void)xtbloom_batch_result_init(&descriptor, sizeof(descriptor));
    descriptor.energies = output_buffer(energies);
    descriptor.forces = output_buffer(qm_forces);
    descriptor.atomic_charges = output_buffer(atomic_charges);
    descriptor.point_charge_forces = output_buffer(point_forces);
    descriptor.scc_iterations = output_buffer(iterations);
    descriptor.scc_converged = output_buffer(converged);
    descriptor.per_system_status = output_buffer(statuses);
  }
};

struct DeviceResult {
  std::vector<double> energies;
  std::vector<double> qm_forces;
  std::vector<double> atomic_charges;
  std::vector<double> point_forces;
  std::vector<std::int32_t> iterations;
  std::vector<std::uint8_t> converged;
  std::vector<xtbloom_status_t> statuses;
  std::vector<std::uint32_t> publication_system_errors;
  std::uint64_t publication_epoch = 0u;
  std::uint32_t publication_plan_error = 0u;
  std::uint64_t numerical_body_count = 0u;
};

int run_public_cpu_reference(xtbloom_context_t* context, PublicHostBatch& batch,
                             const xtbloom_compute_options_t& options, ReferenceResult& result) {
  result.bind(batch);
  const xtbloom_status_t status =
      xtbloom_compute(context, &batch.descriptor, &options, &result.descriptor);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return __LINE__;
  }
  for (std::size_t system = 0; system < result.statuses.size(); ++system) {
    if (result.statuses[system] != XTBLOOM_STATUS_SUCCESS || result.converged[system] != 1u ||
        result.iterations[system] <= 0 || result.iterations[system] > options.max_scc_iterations) {
      std::fprintf(stderr,
                   "CPU reference did not converge in %s system=%zu status=%d converged=%u "
                   "iterations=%d\n",
                   g_scenario, system, result.statuses[system],
                   static_cast<unsigned>(result.converged[system]), result.iterations[system]);
      return __LINE__;
    }
  }
  return 0;
}

int download_device_result(const Gfn2CudaExecutionIdentity& identity, cudaStream_t stream,
                           DeviceResult& result) {
  const std::size_t batch = static_cast<std::size_t>(identity.batch_size);
  const std::size_t atoms = static_cast<std::size_t>(identity.total_atoms);
  const std::size_t points = static_cast<std::size_t>(identity.total_point_charges);
  result.energies.resize(batch);
  result.qm_forces.resize(3u * atoms);
  result.atomic_charges.resize(atoms);
  result.point_forces.resize(3u * points);
  result.iterations.resize(batch);
  result.converged.resize(batch);
  result.statuses.resize(batch);
  result.publication_system_errors.resize(batch);

  CHECK(identity.inference_energies != 0u);
  CHECK(identity.inference_qm_forces != 0u);
  CHECK(identity.inference_atomic_charges != 0u);
  CHECK((identity.inference_point_forces != 0u) == (points != 0u));
  CHECK(identity.inference_iterations != 0u);
  CHECK(identity.inference_converged != 0u);
  CHECK(identity.inference_system_statuses != 0u);
  CHECK(identity.inference_publication_epoch_snapshot != 0u);
  CHECK(identity.inference_publication_system_errors != 0u);
  CHECK(identity.inference_publication_plan_error != 0u);
  CHECK(identity.scc_conditional_graph_ready == 1u);
  CHECK(identity.scc_loop_fallback_reason == 0u);
  CHECK(identity.scc_loop_numerical_body_count != 0u);

  CUDA_CHECK(cudaMemcpyAsync(
      result.energies.data(), reinterpret_cast<const void*>(identity.inference_energies),
      result.energies.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(
      result.qm_forces.data(), reinterpret_cast<const void*>(identity.inference_qm_forces),
      result.qm_forces.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(result.atomic_charges.data(),
                             reinterpret_cast<const void*>(identity.inference_atomic_charges),
                             result.atomic_charges.size() * sizeof(double), cudaMemcpyDeviceToHost,
                             stream));
  if (!result.point_forces.empty()) {
    CUDA_CHECK(cudaMemcpyAsync(
        result.point_forces.data(), reinterpret_cast<const void*>(identity.inference_point_forces),
        result.point_forces.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
  }
  CUDA_CHECK(cudaMemcpyAsync(
      result.iterations.data(), reinterpret_cast<const void*>(identity.inference_iterations),
      result.iterations.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(
      result.converged.data(), reinterpret_cast<const void*>(identity.inference_converged),
      result.converged.size() * sizeof(std::uint8_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(
      result.statuses.data(), reinterpret_cast<const void*>(identity.inference_system_statuses),
      result.statuses.size() * sizeof(xtbloom_status_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(
      cudaMemcpyAsync(&result.publication_epoch,
                      reinterpret_cast<const void*>(identity.inference_publication_epoch_snapshot),
                      sizeof(result.publication_epoch), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(
      cudaMemcpyAsync(result.publication_system_errors.data(),
                      reinterpret_cast<const void*>(identity.inference_publication_system_errors),
                      result.publication_system_errors.size() * sizeof(std::uint32_t),
                      cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(
      cudaMemcpyAsync(&result.publication_plan_error,
                      reinterpret_cast<const void*>(identity.inference_publication_plan_error),
                      sizeof(result.publication_plan_error), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(&result.numerical_body_count,
                             reinterpret_cast<const void*>(identity.scc_loop_numerical_body_count),
                             sizeof(result.numerical_body_count), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return 0;
}

bool near(double actual, double expected, double absolute_tolerance,
          double relative_tolerance) noexcept {
  if (!std::isfinite(actual) || !std::isfinite(expected)) return false;
  return std::abs(actual - expected) <=
         absolute_tolerance + relative_tolerance * std::max(std::abs(actual), std::abs(expected));
}

int compare_values(const char* quantity, const std::vector<double>& actual,
                   const std::vector<double>& expected, double absolute_tolerance,
                   double relative_tolerance) {
  CHECK(actual.size() == expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!near(actual[index], expected[index], absolute_tolerance, relative_tolerance)) {
      std::fprintf(stderr,
                   "%s mismatch in %s index=%zu CUDA=%.17g CPU=%.17g abs_diff=%.6e "
                   "atol=%.3e rtol=%.3e\n",
                   quantity, g_scenario, index, actual[index], expected[index],
                   std::abs(actual[index] - expected[index]), absolute_tolerance,
                   relative_tolerance);
      return __LINE__;
    }
  }
  return 0;
}

int compare_device_to_reference(const DeviceResult& actual, const ReferenceResult& expected,
                                std::uint64_t expected_epoch,
                                const xtbloom_compute_options_t& options) {
  CHECK(actual.statuses.size() == expected.statuses.size());
  CHECK(actual.publication_epoch == expected_epoch);
  CHECK(actual.publication_plan_error ==
        static_cast<std::uint32_t>(Gfn2InferencePublicationPlanError::kSuccess));
  for (std::size_t system = 0; system < actual.statuses.size(); ++system) {
    CHECK(actual.statuses[system] == XTBLOOM_STATUS_SUCCESS);
    CHECK(actual.statuses[system] == expected.statuses[system]);
    CHECK(actual.converged[system] == 1u);
    CHECK(actual.iterations[system] > 0);
    CHECK(actual.iterations[system] <= options.max_scc_iterations);
    CHECK(actual.publication_system_errors[system] ==
          static_cast<std::uint32_t>(Gfn2InferencePublicationSystemError::kSuccess));
  }
  int status = compare_values("energy", actual.energies, expected.energies,
                              kEnergyAbsoluteTolerance, kEnergyRelativeTolerance);
  if (status != 0) return status;
  status = compare_values("atomic charge", actual.atomic_charges, expected.atomic_charges,
                          kChargeAbsoluteTolerance, kChargeRelativeTolerance);
  if (status != 0) return status;
  status = compare_values("QM force", actual.qm_forces, expected.qm_forces, kForceAbsoluteTolerance,
                          kForceRelativeTolerance);
  if (status != 0) return status;
  return compare_values("point-charge force", actual.point_forces, expected.point_forces,
                        kForceAbsoluteTolerance, kForceRelativeTolerance);
}

struct Configuration {
  const char* name;
  SmallSystemKind system;
  bool enable_d4;
  bool enable_points;
  bool enable_periodic;
};

constexpr std::array<Configuration, 4> kConfigurations{{
    {"base", SmallSystemKind::kHe, false, false, false},
    {"d4", SmallSystemKind::kH2, true, false, false},
    {"qm_mm", SmallSystemKind::kH2, true, true, false},
    {"periodic", SmallSystemKind::kH2, true, false, true},
}};

HostSccCaseOptions fixture_options(const Configuration& configuration, std::int64_t batch_size) {
  HostSccCaseOptions options{};
  options.systems.assign(static_cast<std::size_t>(batch_size), configuration.system);
  options.enable_d4 = configuration.enable_d4;
  options.enable_explicit_point_charges = configuration.enable_points;
  options.enable_periodic_embedding = configuration.enable_periodic;
  options.maximum_iterations = 64u;
  options.mixer_history = 8;
  options.residual_tolerance = 1.0e-8;
  options.energy_tolerance = 1.0e-8;
  options.electronic_temperature = 0.0;
  return options;
}

/* The public CPU runtime intentionally requires a production LP64 MKL
 * provider. GPU-only test nodes need an equally mathematical but portable
 * reference, so this owner binds the fixture's deterministic tiny-LAPACK
 * backend to the same production GFN2 SCC and terminal-force routines. */
struct TinyReferenceWorkspace {
  AlignedBuffer integral;
  AlignedBuffer eigensolver;
  std::vector<double> es2_matrix;
  std::vector<double> es2_shell;
  std::vector<double> es2_batch;
  std::vector<double> es2_gradient;
  gfn2::ES2Workspace es2{};
  std::vector<double> aes2_pairs;
  std::vector<double> aes2_potential;
  std::vector<double> aes2_batch;
  std::vector<double> aes2_gradient;
  std::vector<double> aes2_coordination;
  gfn2::AES2Workspace aes2{};
  gfn2::EigensolverWorkspace eigensolver_view{};

  bool bind(const HostSccCase& host, std::string& error) {
    const auto& integrals = host.integral_plan();
    const auto& es2_plan = host.es2_plan();
    const auto& aes2_plan = host.aes2_plan();
    const auto& eigensolver_plan = host.eigensolver_plan();
    if (!integral.allocate(integrals.workspace_size_bytes) ||
        !eigensolver.allocate(eigensolver_plan.workspace_size_bytes())) {
      error = "failed to allocate portable CPU reference workspace";
      return false;
    }
    es2_matrix.resize(static_cast<std::size_t>(es2_plan.total_matrix_elements()));
    es2_shell.resize(static_cast<std::size_t>(es2_plan.total_shells()));
    es2_batch.resize(static_cast<std::size_t>(es2_plan.batch_size()));
    es2_gradient.resize(3u * static_cast<std::size_t>(es2_plan.total_atoms()));
    es2 = {es2_matrix.data(),   es2_plan.total_matrix_elements(),
           es2_shell.data(),    es2_plan.total_shells(),
           es2_batch.data(),    es2_plan.batch_size(),
           es2_gradient.data(), 3 * es2_plan.total_atoms()};

    aes2_pairs.resize(static_cast<std::size_t>(aes2_plan.pair_data_elements()));
    aes2_potential.resize(static_cast<std::size_t>(aes2_plan.potential_scratch_elements()));
    aes2_batch.resize(static_cast<std::size_t>(aes2_plan.batch_size()));
    aes2_gradient.resize(static_cast<std::size_t>(aes2_plan.gradient_scratch_elements()));
    aes2_coordination.resize(static_cast<std::size_t>(aes2_plan.coordination_scratch_elements()));
    aes2 = {aes2_pairs.empty() ? nullptr : aes2_pairs.data(),
            aes2_plan.pair_data_elements(),
            aes2_potential.data(),
            aes2_plan.potential_scratch_elements(),
            aes2_batch.data(),
            aes2_plan.batch_size(),
            aes2_gradient.data(),
            aes2_plan.gradient_scratch_elements(),
            aes2_coordination.data(),
            aes2_plan.coordination_scratch_elements()};
    return gfn2::bind_eigensolver_workspace(eigensolver_plan, eigensolver.data(),
                                            eigensolver.size(), eigensolver_view,
                                            error) == XTBLOOM_STATUS_SUCCESS;
  }
};

xtbloom_status_t refresh_tiny_reference_geometry(HostSccCase& host, const PublicHostBatch& batch,
                                                 TinyReferenceWorkspace& workspace,
                                                 gfn2::CoordinationPlan& coordination,
                                                 gfn2::RepulsionPlan& repulsion,
                                                 std::string& error) {
  auto& positions = const_cast<std::vector<double>&>(host.positions());
  auto& coordination_numbers = const_cast<std::vector<double>&>(host.coordination_numbers());
  auto& point_positions = const_cast<std::vector<double>&>(host.point_charge_positions());
  auto& point_charges = const_cast<std::vector<double>&>(host.point_charge_charges());
  auto& point_hardnesses = const_cast<std::vector<double>&>(host.point_charge_hardnesses());
  positions = batch.positions;
  point_positions = batch.point_positions;
  point_charges = batch.point_values;
  point_hardnesses = batch.point_gammas;
  if (!batch.periodic_shifts.empty()) {
    host.periodic_shifts() = batch.periodic_shifts;
    host.periodic_response_matrices() = batch.response_matrix;
  }

  xtbloom_status_t status = gfn2::make_coordination_plan(
      host.batch_size(), host.total_atoms(), host.atom_offsets().data(),
      host.atomic_numbers().data(), coordination, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status =
      gfn2::make_repulsion_plan(host.batch_size(), host.total_atoms(), host.atom_offsets().data(),
                                host.atomic_numbers().data(), repulsion, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = gfn2::evaluate_coordination_cpu(coordination, positions.data(),
                                           coordination_numbers.data(), error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = gfn2::evaluate_overlap_cpu(host.basis_plan(), host.integral_plan(), positions.data(),
                                      host.overlap().data(), workspace.integral.data(),
                                      workspace.integral.size(), error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = gfn2::evaluate_multipole_cpu(
      host.basis_plan(), host.integral_plan(), positions.data(), host.dipole_integrals().data(),
      host.quadrupole_integrals().data(), workspace.integral.data(), workspace.integral.size(),
      error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = gfn2::evaluate_h0_cpu(host.basis_plan(), host.integral_plan(), host.h0_plan(),
                                 positions.data(), coordination_numbers.data(),
                                 host.overlap().data(), host.h0().data(), error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  const std::uint64_t generation = host.options().geometry_generation;
  auto& es2_cache = host.es2_cache();
  status = gfn2::update_es2_geometry_cache_cpu(
      host.es2_plan(), positions.data(), generation, es2_cache.coulomb_matrix,
      static_cast<std::size_t>(es2_cache.matrix_elements), workspace.es2, es2_cache, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  auto& aes2_cache = host.aes2_cache();
  status = gfn2::update_aes2_geometry_cache_cpu(
      host.aes2_plan(), positions.data(), coordination_numbers.data(), generation,
      aes2_cache.pair_data, static_cast<std::size_t>(aes2_cache.pair_data_elements), workspace.aes2,
      aes2_cache, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  if (host.point_charge_plan() != nullptr) {
    status = gfn2::evaluate_external_point_charge_potential_cpu(
        *host.point_charge_plan(), positions.data(), point_positions.data(), point_charges.data(),
        point_hardnesses.data(), host.explicit_point_charge_shell_potential().data(), error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  }
  if (host.d4_plan() != nullptr) {
    auto* d4_cache = host.d4_cache();
    status = gfn2::update_d4_geometry_cache_cpu(
        *host.d4_plan(), positions.data(), generation, d4_cache->pair_data,
        static_cast<std::size_t>(d4_cache->pair_data_elements), d4_cache->coordination_numbers,
        static_cast<std::size_t>(d4_cache->coordination_elements),
        host.driver_workspace().d4_workspace, *d4_cache, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  }
  auto& overlap_cache = const_cast<gfn2::EigensolverOverlapCache&>(host.overlap_cache());
  status = gfn2::factor_overlap_cpu(host.eigensolver_plan(), host.overlap().data(), generation,
                                    host.cpu_backend(), workspace.eigensolver_view, overlap_cache,
                                    error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  auto& geometry = host.geometry();
  geometry.h0 = host.h0().data();
  geometry.integrals = {
      host.overlap().data(), host.dipole_integrals().data(), host.quadrupole_integrals().data(),
      host.integral_plan().total_matrix_elements, host.mulliken_plan().identity()};
  geometry.es2_cache = host.es2_cache();
  geometry.aes2_cache = host.aes2_cache();
  geometry.geometry_generation = generation;
  if (host.d4_cache() != nullptr) geometry.d4_cache = *host.d4_cache();
  if (host.periodic_plan() != nullptr) {
    geometry.periodic_shifts = host.periodic_shifts().data();
    geometry.periodic_response_matrices = host.periodic_response_matrices().data();
    geometry.periodic_embedding_generation = generation;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t compose_tiny_reference(HostSccCase& host, const PublicHostBatch& batch,
                                        TinyReferenceWorkspace& workspace,
                                        const gfn2::CoordinationPlan& coordination,
                                        const gfn2::RepulsionPlan& repulsion,
                                        ReferenceResult& result, std::string& error) {
  const std::size_t batch_size = static_cast<std::size_t>(host.batch_size());
  const std::size_t atoms = static_cast<std::size_t>(host.total_atoms());
  const std::size_t shells = static_cast<std::size_t>(host.wavefunction_layout().total_shells);
  const std::size_t matrices = static_cast<std::size_t>(host.integral_plan().total_matrix_elements);
  const std::size_t points = batch.point_values.size();
  std::vector<double> component_shell(shells);
  std::vector<double> scalar_shell(shells);
  std::vector<double> atomic_potential(atoms);
  std::vector<double> d4_atomic_potential(atoms, 0.0);
  std::vector<double> periodic_atomic_potential(atoms, 0.0);
  std::vector<double> dipole_potential(3u * atoms);
  std::vector<double> quadrupole_potential(6u * atoms);
  std::vector<double> component_energy(batch_size);

  xtbloom_status_t status =
      gfn2::evaluate_es2_potential_cpu(host.es2_plan(), host.es2_cache(), host.wavefunction().qsh,
                                       component_shell.data(), workspace.es2, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  scalar_shell = component_shell;
  status = gfn2::evaluate_es3_potential_cpu(gfn2::make_es3_view(host.es3_plan()),
                                            host.wavefunction().qsh, component_shell.data(), error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  for (std::size_t shell = 0; shell < shells; ++shell) {
    scalar_shell[shell] += component_shell[shell];
    if (host.point_charge_plan() != nullptr) {
      scalar_shell[shell] += host.explicit_point_charge_shell_potential()[shell];
    }
  }
  status = gfn2::evaluate_aes2_potential_cpu(
      host.aes2_plan(), host.aes2_cache(), host.wavefunction().qat, host.wavefunction().dipole,
      host.wavefunction().quadrupole, atomic_potential.data(), dipole_potential.data(),
      quadrupole_potential.data(), workspace.aes2, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (host.d4_plan() != nullptr) {
    status = gfn2::evaluate_d4_two_body_cpu(
        *host.d4_plan(), *host.d4_cache(), host.wavefunction().qat, component_energy.data(),
        d4_atomic_potential.data(), host.driver_workspace().d4_workspace, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  }
  if (host.periodic_plan() != nullptr) {
    std::vector<double> periodic_energy(batch_size);
    std::vector<xtbloom_status_t> periodic_status(batch_size);
    const gfn2::PeriodicEmbeddingView view{
        host.periodic_shifts().data(),
        static_cast<std::int64_t>(host.periodic_shifts().size()),
        host.periodic_response_matrices().data(),
        static_cast<std::int64_t>(host.periodic_response_matrices().size()),
        host.wavefunction().qat,
        static_cast<std::int64_t>(atoms),
        periodic_atomic_potential.data(),
        static_cast<std::int64_t>(atoms),
        periodic_energy.data(),
        static_cast<std::int64_t>(batch_size),
        periodic_status.data(),
        static_cast<std::int64_t>(batch_size),
        host.periodic_plan()->identity(),
    };
    status = gfn2::evaluate_periodic_embedding_batch_cpu(
        *host.periodic_plan(), view, host.driver_workspace().periodic_embedding_workspace, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  }
  for (std::size_t atom = 0; atom < atoms; ++atom) {
    atomic_potential[atom] += d4_atomic_potential[atom] + periodic_atomic_potential[atom];
  }
  for (std::size_t shell = 0; shell < shells; ++shell) {
    scalar_shell[shell] +=
        atomic_potential[static_cast<std::size_t>(host.basis_plan().shell_to_atom[shell])];
  }

  std::vector<double> energy_scratch(batch_size);
  std::vector<double> total_gradient(3u * atoms);
  std::vector<double> component_gradient(3u * atoms);
  std::vector<double> force_scratch(3u * atoms);
  std::vector<double> point_force_scratch(3u * points);
  std::vector<double> overlap_adjoint(matrices);
  std::vector<double> dipole_adjoint(3u * matrices);
  std::vector<double> quadrupole_adjoint(6u * matrices);
  std::vector<double> coordination_adjoint(atoms);
  gfn2::RestrictedGfn2ForceWorkspace force_workspace{
      energy_scratch.data(),
      host.d4_plan() == nullptr ? nullptr : component_energy.data(),
      static_cast<std::int64_t>(batch_size),
      total_gradient.data(),
      component_gradient.data(),
      force_scratch.data(),
      static_cast<std::int64_t>(3u * atoms),
      overlap_adjoint.data(),
      static_cast<std::int64_t>(matrices),
      dipole_adjoint.data(),
      static_cast<std::int64_t>(3u * matrices),
      quadrupole_adjoint.data(),
      static_cast<std::int64_t>(6u * matrices),
      coordination_adjoint.data(),
      static_cast<std::int64_t>(atoms),
      point_force_scratch.empty() ? nullptr : point_force_scratch.data(),
      static_cast<std::int64_t>(point_force_scratch.size()),
      workspace.integral.data(),
      workspace.integral.size(),
      workspace.es2,
      workspace.aes2,
      host.d4_plan() == nullptr ? gfn2::D4Workspace{} : host.driver_workspace().d4_workspace,
  };
  const gfn2::RestrictedGfn2StationaryInput input{
      batch.positions.data(),
      host.coordination_numbers().data(),
      host.options().geometry_generation,
      host.overlap().data(),
      host.wavefunction().density,
      host.wavefunction().energy_weighted_density,
      host.wavefunction().qsh,
      host.wavefunction().qat,
      host.wavefunction().dipole,
      host.wavefunction().quadrupole,
      scalar_shell.data(),
      dipole_potential.data(),
      quadrupole_potential.data(),
      host.driver_state().free_energies,
      points == 0u ? nullptr : batch.point_positions.data(),
      points == 0u ? nullptr : batch.point_values.data(),
      points == 0u ? nullptr : batch.point_gammas.data(),
  };
  gfn2::AES2GeometryCache force_aes2_cache = host.aes2_cache();
  if (force_aes2_cache.pair_data_elements == 0) {
    /* HostSccCase owns an aligned one-byte placeholder for zero-size caches;
     * the public force contract requires the canonical null/zero descriptor. */
    force_aes2_cache.pair_data = nullptr;
  }
  status = gfn2::evaluate_restricted_gfn2_energy_forces_cpu(
      host.basis_plan(), host.integral_plan(), coordination, repulsion, host.h0_plan(),
      host.mulliken_plan(), host.es2_plan(), host.es2_cache(), host.aes2_plan(), force_aes2_cache,
      host.d4_plan(), host.d4_cache(), host.point_charge_plan(), input, result.energies.data(),
      result.qm_forces.data(), result.point_forces.empty() ? nullptr : result.point_forces.data(),
      {}, force_workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  std::copy_n(host.wavefunction().qat, atoms, result.atomic_charges.data());
  for (std::size_t system = 0; system < batch_size; ++system) {
    result.iterations[system] = static_cast<std::int32_t>(std::min<std::uint64_t>(
        host.driver_state().iterations[system],
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())));
    result.converged[system] = host.driver_state().converged[system];
    result.statuses[system] = host.driver_state().system_statuses[system];
  }
  return XTBLOOM_STATUS_SUCCESS;
}

int run_tiny_cpu_reference(const Configuration& configuration, PublicHostBatch& batch,
                           ReferenceResult& result) {
  HostSccCase host;
  std::string error;
  xtbloom_status_t status =
      HostSccCase::create(fixture_options(configuration, batch.descriptor.batch_size), host, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "portable CPU fixture failed in %s: status=%d error=%s\n", g_scenario,
                 static_cast<int>(status), error.c_str());
    return __LINE__;
  }
  TinyReferenceWorkspace workspace;
  if (!workspace.bind(host, error)) {
    std::fprintf(stderr, "portable CPU workspace failed in %s: %s\n", g_scenario, error.c_str());
    return __LINE__;
  }
  gfn2::CoordinationPlan coordination;
  gfn2::RepulsionPlan repulsion;
  status = refresh_tiny_reference_geometry(host, batch, workspace, coordination, repulsion, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "portable CPU geometry refresh failed in %s: status=%d error=%s\n",
                 g_scenario, static_cast<int>(status), error.c_str());
    return __LINE__;
  }
  for (std::uint64_t iteration = 0u; iteration < host.options().maximum_iterations; ++iteration) {
    status = host.run_one_iteration(error);
    if (status != XTBLOOM_STATUS_SUCCESS && status != XTBLOOM_STATUS_SCC_NOT_CONVERGED &&
        status != XTBLOOM_STATUS_EIGENSOLVER_FAILED) {
      std::fprintf(stderr, "portable CPU SCC failed in %s iteration=%llu status=%d error=%s\n",
                   g_scenario, static_cast<unsigned long long>(iteration), static_cast<int>(status),
                   error.c_str());
      return __LINE__;
    }
  }
  result.bind(batch);
  for (std::size_t system = 0; system < result.statuses.size(); ++system) {
    if (host.driver_state().system_statuses[system] != XTBLOOM_STATUS_SUCCESS ||
        host.driver_state().converged[system] != 1u) {
      std::fprintf(stderr,
                   "portable CPU reference did not converge in %s system=%zu status=%d "
                   "converged=%u\n",
                   g_scenario, system,
                   static_cast<int>(host.driver_state().system_statuses[system]),
                   static_cast<unsigned>(host.driver_state().converged[system]));
      return __LINE__;
    }
  }
  status = compose_tiny_reference(host, batch, workspace, coordination, repulsion, result, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "portable CPU terminal composition failed in %s: status=%d error=%s\n",
                 g_scenario, static_cast<int>(status), error.c_str());
    return __LINE__;
  }
  return 0;
}

int run_cpu_reference(xtbloom_context_t* context, const Configuration& configuration,
                      PublicHostBatch& batch, const xtbloom_compute_options_t& options,
                      ReferenceResult& result) {
  const int public_status = run_public_cpu_reference(context, batch, options, result);
  if (public_status == 0) return 0;
  const char* public_error = xtbloom_get_last_error();
  if (public_error == nullptr || std::strstr(public_error, "failed to load libmkl_rt") == nullptr) {
    std::fprintf(stderr, "CPU reference failed in %s: error=%s\n", g_scenario,
                 public_error == nullptr ? "unknown public CPU failure" : public_error);
    return public_status;
  }
  /* A missing optional production BLAS provider is an environment property,
   * not grounds to skip CUDA correctness. Recompute through the fixture's
   * deterministic LP64 backend and the same production model functions. */
  return run_tiny_cpu_reference(configuration, batch, result);
}

int execute_and_compare(Gfn2CudaExecutionCache& cache, cudaStream_t stream,
                        Gfn2CudaSccStartMode mode, const ReferenceResult& reference,
                        std::uint64_t expected_epoch, const xtbloom_compute_options_t& options) {
  std::string error;
  const xtbloom_status_t status = cache.execute_inference_async(mode, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "CUDA inference submission failed in %s: status=%d error=%s\n", g_scenario,
                 static_cast<int>(status), error.c_str());
    return __LINE__;
  }
  DeviceResult result;
  int line = download_device_result(cache.identity(), stream, result);
  if (line != 0) return line;
  CHECK(!result.iterations.empty());
  const std::int32_t maximum_peer_iterations =
      *std::max_element(result.iterations.begin(), result.iterations.end());
  CHECK(maximum_peer_iterations > 0);
  /* Each WHILE body publishes exactly one attempted peer iteration. The loop
   * therefore stops after the slowest active peer becomes terminal. */
  CHECK(result.numerical_body_count == static_cast<std::uint64_t>(maximum_peer_iterations));
  return compare_device_to_reference(result, reference, expected_epoch, options);
}

int run_matrix_member(xtbloom_context_t* cpu_context, cudaStream_t stream, std::int32_t device_id,
                      const Configuration& configuration, std::int64_t batch_size) {
  std::string scenario = std::string(configuration.name) +
                         "/batch=" + std::to_string(static_cast<long long>(batch_size)) + "/setup";
  g_scenario = scenario.c_str();

  HostSccCase fixture;
  std::string error;
  CHECK(HostSccCase::create(fixture_options(configuration, batch_size), fixture, error) ==
        XTBLOOM_STATUS_SUCCESS);
  PublicHostBatch batch = PublicHostBatch::from_fixture(fixture, configuration.enable_periodic);
  const xtbloom_compute_options_t options = compute_options();

  ReferenceResult initial_reference;
  scenario = std::string(configuration.name) +
             "/batch=" + std::to_string(static_cast<long long>(batch_size)) + "/cpu-initial";
  g_scenario = scenario.c_str();
  int line = run_cpu_reference(cpu_context, configuration, batch, options, initial_reference);
  if (line != 0) return line;

  Gfn2CudaExecutionCache cache(device_id, reinterpret_cast<void*>(stream));
  bool reused = true;
  const xtbloom_status_t setup_status =
      cache.prepare_host(batch.descriptor, options, reused, error);
  if (setup_status != XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "CUDA setup failed in %s: status=%d error=%s\n", g_scenario,
                 static_cast<int>(setup_status), error.c_str());
    return __LINE__;
  }
  CHECK(!reused);
  CHECK(cache.identity().batch_size == batch_size);
  CHECK(cache.identity().scc_conditional_graph_ready == 1u);
  const Gfn2CudaExecutionIdentity stable_identity = cache.identity();

  scenario = std::string(configuration.name) +
             "/batch=" + std::to_string(static_cast<long long>(batch_size)) + "/fresh-initial";
  g_scenario = scenario.c_str();
  line = execute_and_compare(cache, stream, Gfn2CudaSccStartMode::kFresh, initial_reference, 1u,
                             options);
  if (line != 0) return line;

  scenario = std::string(configuration.name) +
             "/batch=" + std::to_string(static_cast<long long>(batch_size)) + "/warm-initial";
  g_scenario = scenario.c_str();
  line = execute_and_compare(cache, stream, Gfn2CudaSccStartMode::kWarm, initial_reference, 1u,
                             options);
  if (line != 0) return line;

  batch.perturb_geometry();
  ReferenceResult changed_reference;
  scenario = std::string(configuration.name) +
             "/batch=" + std::to_string(static_cast<long long>(batch_size)) + "/cpu-changed";
  g_scenario = scenario.c_str();
  line = run_cpu_reference(cpu_context, configuration, batch, options, changed_reference);
  if (line != 0) return line;

  const xtbloom_status_t refresh_status =
      cache.prepare_host(batch.descriptor, options, reused, error);
  if (refresh_status != XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "CUDA changed-geometry refresh failed in %s: status=%d error=%s\n",
                 g_scenario, static_cast<int>(refresh_status), error.c_str());
    return __LINE__;
  }
  CHECK(reused);
  const Gfn2CudaExecutionIdentity refreshed_identity = cache.identity();
  CHECK(refreshed_identity.scc_conditional_graph_ready == 1u);
  CHECK(refreshed_identity.scc_loop_owner == stable_identity.scc_loop_owner);
  CHECK(refreshed_identity.scc_loop_active_count == stable_identity.scc_loop_active_count);
  CHECK(refreshed_identity.scc_loop_numerical_body_count ==
        stable_identity.scc_loop_numerical_body_count);

  scenario = std::string(configuration.name) +
             "/batch=" + std::to_string(static_cast<long long>(batch_size)) + "/fresh-changed";
  g_scenario = scenario.c_str();
  line = execute_and_compare(cache, stream, Gfn2CudaSccStartMode::kFresh, changed_reference, 2u,
                             options);
  if (line != 0) return line;

  scenario = std::string(configuration.name) +
             "/batch=" + std::to_string(static_cast<long long>(batch_size)) + "/warm-changed";
  g_scenario = scenario.c_str();
  return execute_and_compare(cache, stream, Gfn2CudaSccStartMode::kWarm, changed_reference, 2u,
                             options);
}

}  // namespace

int main() {
  int device_count = 0;
  cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
  if (cuda_status == cudaErrorNoDevice || device_count == 0) {
    std::puts("cuda_gfn2_runtime_parity_test: SKIP (no CUDA device)");
    return 0;
  }
  if (cuda_status != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(cuda_status));
    return 1;
  }

  int device = 0;
  cuda_status = cudaGetDevice(&device);
  if (cuda_status != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDevice failed: %s\n", cudaGetErrorString(cuda_status));
    return 1;
  }
  cudaStream_t stream = nullptr;
  cuda_status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (cuda_status != cudaSuccess) {
    std::fprintf(stderr, "cudaStreamCreateWithFlags failed: %s\n", cudaGetErrorString(cuda_status));
    return 1;
  }

  CpuContext cpu_context = make_cpu_context();
  if (cpu_context == nullptr) {
    std::fprintf(stderr, "failed to create CPU reference context: %s\n", xtbloom_get_last_error());
    (void)cudaStreamDestroy(stream);
    return 1;
  }

  int result = 0;
  for (const Configuration& configuration : kConfigurations) {
    for (const std::int64_t batch_size : kBatchSizes) {
      result = run_matrix_member(cpu_context.get(), stream, device, configuration, batch_size);
      if (result != 0) break;
    }
    if (result != 0) break;
  }

  const cudaError_t destroy_status = cudaStreamDestroy(stream);
  if (result == 0 && destroy_status != cudaSuccess) {
    std::fprintf(stderr, "cudaStreamDestroy failed: %s\n", cudaGetErrorString(destroy_status));
    return 1;
  }
  if (result == 0) {
    std::puts("cuda_gfn2_runtime_parity_test: PASS");
  }
  return result;
}
