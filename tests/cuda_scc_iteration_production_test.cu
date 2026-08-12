#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "backends/cuda/gfn2_scc_iteration_arena.cuh"
#include "backends/cuda/gfn2_scc_iteration_initialize.cuh"
#include "backends/cuda/gfn2_scc_iteration_reports.cuh"
#include "backends/cuda/gfn2_scc_loop.cuh"
#include "backends/cuda/gfn2_scc_setup_eigensolver.cuh"
#include "backends/cuda/gfn2_scc_setup_inputs.cuh"
#include "backends/cuda/gfn2_scc_setup_topology.hpp"
#include "data/parameters/d4.hpp"
#include "model/gfn2/coordination.hpp"
#include "runtime/nvidia_host_api.h"
#include "tests/support/cuda_d4_pairlist_fixture.cuh"
#include "tests/support/gfn2_scc_test_case.hpp"

#define CHECK(condition)                                                                          \
  do {                                                                                            \
    if (!(condition)) {                                                                           \
      std::fprintf(stderr, "production SCC check failed at line %d: %s\n", __LINE__, #condition); \
      return __LINE__;                                                                            \
    }                                                                                             \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using namespace xtbloom::detail;
using namespace xtbloom::detail::cuda;
using xtbloom::test::gfn2::HostSccCase;
using xtbloom::test::gfn2::HostSccCaseOptions;
using xtbloom::test::gfn2::HostSccCheckpoint;
using xtbloom::test::gfn2::SmallSystemKind;

constexpr std::uint64_t kPlanToken = 0x105105105ULL;
constexpr std::uint64_t kGeometryGeneration = 105u;
constexpr std::uint64_t kInitializationGeneration = 1u;

struct CouplingSelection {
  bool d4 = false;
  bool point_charges = false;
  bool periodic = false;
};

template <typename T>
Gfn2SccSetupHostArray<T> setup_view(const std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), static_cast<std::int64_t>(values.size())};
}

template <typename T>
Gfn2SccIterationHostArrayView<T> initialization_view(const T* values,
                                                     std::int64_t elements) noexcept {
  return {elements == 0 ? nullptr : values, elements};
}

class DeviceAllocation {
 public:
  DeviceAllocation() = default;
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  ~DeviceAllocation() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
  }

  bool allocate(std::size_t bytes) noexcept {
    bytes_ = bytes;
    return bytes != 0u && cudaMalloc(&pointer_, bytes) == cudaSuccess;
  }

  void* get() const noexcept { return pointer_; }
  std::size_t bytes() const noexcept { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0u;
};

class PinnedAllocation {
 public:
  PinnedAllocation() = default;
  PinnedAllocation(const PinnedAllocation&) = delete;
  PinnedAllocation& operator=(const PinnedAllocation&) = delete;

  ~PinnedAllocation() {
    if (pointer_ != nullptr) {
      (void)cudaFreeHost(pointer_);
    }
  }

  bool allocate(std::size_t bytes) noexcept {
    bytes_ = bytes;
    return bytes == 0u || cudaMallocHost(&pointer_, bytes) == cudaSuccess;
  }

  void* get() const noexcept { return pointer_; }
  std::size_t bytes() const noexcept { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0u;
};

class GraphResources {
 public:
  GraphResources() = default;
  GraphResources(const GraphResources&) = delete;
  GraphResources& operator=(const GraphResources&) = delete;

  ~GraphResources() {
    if (executable_ != nullptr) {
      (void)cudaGraphExecDestroy(executable_);
    }
    if (graph_ != nullptr) {
      (void)cudaGraphDestroy(graph_);
    }
  }

  cudaGraph_t* graph_address() noexcept { return &graph_; }
  cudaGraph_t graph() const noexcept { return graph_; }
  cudaGraphExec_t* executable_address() noexcept { return &executable_; }
  cudaGraphExec_t executable() const noexcept { return executable_; }

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
};

class ProviderHandles {
 public:
  ProviderHandles() = default;
  ProviderHandles(const ProviderHandles&) = delete;
  ProviderHandles& operator=(const ProviderHandles&) = delete;

  ~ProviderHandles() {
    if (blas_ != nullptr) {
      (void)cublasDestroy(blas_);
    }
    if (parameters_ != nullptr) {
      (void)cusolverDnDestroyParams(parameters_);
    }
    if (solver_ != nullptr) {
      (void)cusolverDnDestroy(solver_);
    }
    if (stream_ != nullptr) {
      (void)cudaStreamSynchronize(stream_);
      (void)cudaStreamDestroy(stream_);
    }
  }

  bool create() noexcept {
    return cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) == cudaSuccess &&
           cusolverDnCreate(&solver_) == CUSOLVER_STATUS_SUCCESS &&
           cusolverDnCreateParams(&parameters_) == CUSOLVER_STATUS_SUCCESS &&
           cublasCreate(&blas_) == CUBLAS_STATUS_SUCCESS;
  }

  cudaStream_t stream() const noexcept { return stream_; }
  cusolverDnHandle_t solver() const noexcept { return solver_; }
  cusolverDnParams_t parameters() const noexcept { return parameters_; }
  cublasHandle_t blas() const noexcept { return blas_; }

 private:
  cudaStream_t stream_ = nullptr;
  cusolverDnHandle_t solver_ = nullptr;
  cusolverDnParams_t parameters_ = nullptr;
  cublasHandle_t blas_ = nullptr;
};

/* Establish setup-stream ordering for a distinct nonblocking execution
 * stream without synchronizing the host. The provider handles and mutable SCC
 * arena remain single-flight for the complete lifetime of this object. */
class OrderedExecutionStream {
 public:
  OrderedExecutionStream() = default;
  OrderedExecutionStream(const OrderedExecutionStream&) = delete;
  OrderedExecutionStream& operator=(const OrderedExecutionStream&) = delete;

  ~OrderedExecutionStream() {
    if (stream_ != nullptr) {
      (void)cudaStreamSynchronize(stream_);
    }
    if (event_ != nullptr) {
      (void)cudaEventDestroy(event_);
    }
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
    }
  }

  bool create_after(cudaStream_t setup_stream) noexcept {
    return cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) == cudaSuccess &&
           cudaEventCreateWithFlags(&event_, cudaEventDisableTiming) == cudaSuccess &&
           cudaEventRecord(event_, setup_stream) == cudaSuccess &&
           cudaStreamWaitEvent(stream_, event_, 0u) == cudaSuccess;
  }

  cudaStream_t stream() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
  cudaEvent_t event_ = nullptr;
};

/* HostSccCase exposes the production component caches but not the common
 * geometry-pair cache. The SCC iteration only validates that common cache in
 * this energy-only path; geometry VJPs consume it later. A deterministic
 * finite image therefore supplies the production owner until the host fixture
 * grows a common geometry-cache accessor. */
struct InputBacking {
  xtbloom::detail::gfn2::CoordinationPlan coordination_plan;
  std::vector<double> geometry_pair_data;
  std::vector<std::uint64_t> geometry_generations;
  std::vector<Gfn2D4DeviceElementData> d4_elements;
  std::vector<Gfn2D4DeviceReferenceData> d4_references;

  bool prepare(const HostSccCase& host, std::string& error) {
    if (xtbloom::detail::gfn2::make_coordination_plan(
            host.batch_size(), host.total_atoms(), host.atom_offsets().data(),
            host.atomic_numbers().data(), coordination_plan, error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    geometry_pair_data.resize(static_cast<std::size_t>(host.aes2_plan().total_pairs()) *
                              kGfn2GeometryPairDataElements);
    for (std::size_t index = 0; index < geometry_pair_data.size(); ++index) {
      geometry_pair_data[index] = 0.001 * static_cast<double>(index + 1u);
    }
    geometry_generations.assign(static_cast<std::size_t>(host.batch_size()),
                                host.options().geometry_generation);

    d4_elements.reserve(xtbloom::parameters::d4::kElements.size());
    for (const auto& element : xtbloom::parameters::d4::kElements) {
      d4_elements.push_back({element.reference_offset, element.reference_count,
                             element.covalent_radius, element.electronegativity,
                             element.effective_charge, element.hardness, element.r4r2});
    }
    d4_references.reserve(xtbloom::parameters::d4::kReferences.size());
    for (const auto& reference : xtbloom::parameters::d4::kReferences) {
      d4_references.push_back(
          {reference.coordination_number, reference.charge, reference.gaussian_count});
    }
    return true;
  }

  Gfn2SccSetupInputSources sources(const HostSccCase& host) const noexcept {
    Gfn2SccSetupInputSources result{};
    result.basis = &host.basis_plan();
    result.integrals = &host.integral_plan();
    result.h0_plan = &host.h0_plan();
    result.wavefunction = &host.wavefunction_layout();
    result.es2 = &host.es2_plan();
    result.es3 = &host.es3_plan();
    result.aes2 = &host.aes2_plan();
    result.mulliken = &host.mulliken_plan();
    result.mixer = &host.mixer_plan();
    result.driver = &host.driver_plan();
    result.geometry_generation = host.options().geometry_generation;
    result.atomic_numbers = setup_view(host.atomic_numbers());
    result.positions = setup_view(host.positions());
    result.covalent_radii = setup_view(coordination_plan.covalent_radius);
    result.h0 = setup_view(host.h0());
    result.overlap = setup_view(host.overlap());
    result.dipole_integrals = setup_view(host.dipole_integrals());
    result.quadrupole_integrals = setup_view(host.quadrupole_integrals());
    result.geometry_cache.pair_data = setup_view(geometry_pair_data);
    result.geometry_cache.coordination_numbers = setup_view(host.coordination_numbers());
    result.geometry_cache.system_generations = setup_view(geometry_generations);
    result.es2_cache.coulomb_matrix = {host.es2_cache().coulomb_matrix,
                                       host.es2_cache().matrix_elements};
    result.aes2_cache.pair_data = {host.aes2_cache().pair_data,
                                   host.aes2_cache().pair_data_elements};

    if (host.d4_plan() != nullptr) {
      result.d4.plan = host.d4_plan();
      result.d4.elements = setup_view(d4_elements);
      result.d4.references = setup_view(d4_references);
      result.d4.reference_c6 = {
          xtbloom::parameters::d4::kReferenceC6.data(),
          static_cast<std::int64_t>(xtbloom::parameters::d4::kReferenceC6.size())};
      result.d4.coordination_numbers = {host.d4_cache()->coordination_numbers,
                                        host.d4_cache()->coordination_elements};
    }
    if (host.point_charge_plan() != nullptr) {
      result.point_charges.plan = host.point_charge_plan();
      result.point_charges.positions = setup_view(host.point_charge_positions());
      result.point_charges.charges = setup_view(host.point_charge_charges());
      result.point_charges.hardnesses = setup_view(host.point_charge_hardnesses());
      result.point_charges.shell_potential_cache =
          setup_view(host.explicit_point_charge_shell_potential());
    }
    if (host.periodic_plan() != nullptr) {
      result.periodic.plan = host.periodic_plan();
      result.periodic.shifts = setup_view(host.periodic_shifts());
      result.periodic.response_matrices = setup_view(host.periodic_response_matrices());
    }
    return result;
  }
};

Gfn2SccIterationHostInitialization fresh_initialization(const HostSccCase& host) noexcept {
  const auto& layout = host.wavefunction_layout();
  const auto& wavefunction = host.wavefunction();
  Gfn2SccIterationHostInitialization result{};
  result.mode = Gfn2SccIterationInitializationMode::kFresh;
  result.plan_token = kPlanToken;
  result.initialization_generation = kInitializationGeneration;
  result.topology = {
      initialization_view(host.atom_offsets().data(),
                          static_cast<std::int64_t>(host.atom_offsets().size())),
      initialization_view(layout.batch_shell_offsets.data(),
                          static_cast<std::int64_t>(layout.batch_shell_offsets.size())),
      kPlanToken};
  result.wavefunction.plan_token = kPlanToken;
  result.wavefunction.population = {
      initialization_view(wavefunction.qsh, layout.qsh.element_count),
      initialization_view(wavefunction.qat, layout.qat.element_count),
      initialization_view(wavefunction.dipole, layout.dipole.element_count),
      initialization_view(wavefunction.quadrupole, layout.quadrupole.element_count), kPlanToken};
  return result;
}

Gfn2SccIterationHostInitialization fresh_initialization(
    const Gfn2WavefunctionLayoutView& host_layout,
    const xtbloom::detail::gfn2::WavefunctionLayout& layout,
    const xtbloom::detail::gfn2::WavefunctionView& wavefunction) noexcept {
  Gfn2SccIterationHostInitialization result{};
  result.mode = Gfn2SccIterationInitializationMode::kFresh;
  result.plan_token = kPlanToken;
  result.initialization_generation = kInitializationGeneration;
  result.topology = {
      initialization_view(layout.atom_offsets.data(),
                          static_cast<std::int64_t>(layout.atom_offsets.size())),
      initialization_view(layout.batch_shell_offsets.data(),
                          static_cast<std::int64_t>(layout.batch_shell_offsets.size())),
      kPlanToken};
  result.wavefunction.plan_token = kPlanToken;
  result.wavefunction.population = {
      initialization_view(wavefunction.qsh, layout.qsh.element_count),
      initialization_view(wavefunction.qat, layout.qat.element_count),
      initialization_view(wavefunction.dipole, layout.dipole.element_count),
      initialization_view(wavefunction.quadrupole, layout.quadrupole.element_count), kPlanToken};
  /* Mixed-spin initialization must use the setup owner's independent host
   * metadata. The corresponding descriptor in plan_seed points at device
   * memory and must never be dereferenced while packing the host image. */
  result.wavefunction_layout = host_layout;
  return result;
}

template <typename T>
bool download(const T* device, std::int64_t elements, std::vector<T>& host, cudaStream_t stream) {
  if (elements < 0) {
    return false;
  }
  host.resize(static_cast<std::size_t>(elements));
  return elements == 0 || cudaMemcpyAsync(host.data(), device, host.size() * sizeof(T),
                                          cudaMemcpyDeviceToHost, stream) == cudaSuccess;
}

template <typename T>
bool download_value(const T* device, T& host, cudaStream_t stream) {
  return device != nullptr &&
         cudaMemcpyAsync(&host, device, sizeof(T), cudaMemcpyDeviceToHost, stream) == cudaSuccess;
}

template <typename T>
bool upload_fill(T* device, std::int64_t elements, T value) {
  if (device == nullptr || elements < 0) {
    return false;
  }
  std::vector<T> host(static_cast<std::size_t>(elements), value);
  return elements == 0 || cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                                     cudaMemcpyHostToDevice) == cudaSuccess;
}

bool near(double first, double second, double tolerance) noexcept {
  const double scale = std::max({1.0, std::abs(first), std::abs(second)});
  return std::abs(first - second) <= tolerance * scale;
}

bool compare_doubles(const char* field, const std::vector<double>& actual, const double* expected,
                     std::int64_t elements, double tolerance) {
  if (expected == nullptr || elements < 0 || actual.size() != static_cast<std::size_t>(elements)) {
    std::fprintf(stderr, "%s has an invalid parity extent\n", field);
    return false;
  }
  for (std::int64_t index = 0; index < elements; ++index) {
    if (!near(actual[static_cast<std::size_t>(index)], expected[index], tolerance)) {
      std::fprintf(stderr, "%s mismatch at %lld: CUDA=%.17g CPU=%.17g delta=%.3e tolerance=%.3e\n",
                   field, static_cast<long long>(index), actual[static_cast<std::size_t>(index)],
                   expected[index], actual[static_cast<std::size_t>(index)] - expected[index],
                   tolerance);
      return false;
    }
  }
  return true;
}

/* Generalized eigenvectors are unique only up to sign for isolated roots and
 * up to an orthogonal rotation within a degenerate eigenspace. Compare those
 * two invariants separately so LAPACK/cuSOLVER provider choices cannot make a
 * scientifically equivalent solution fail parity. */
bool compare_coefficients(const HostSccCase& host, const std::vector<double>& actual_eigenvalues,
                          std::vector<double> actual, double eigenvalue_tolerance,
                          double coefficient_tolerance) {
  const auto& layout = host.wavefunction_layout();
  const double* expected_eigenvalues = host.wavefunction().eigenvalues;
  const double* expected = host.wavefunction().coefficients;
  if (actual_eigenvalues.size() != static_cast<std::size_t>(layout.eigenvalues.element_count) ||
      actual.size() != static_cast<std::size_t>(layout.coefficients.element_count)) {
    std::fprintf(stderr, "eigenpairs have an invalid parity extent\n");
    return false;
  }
  for (std::int64_t system = 0; system < layout.batch_size; ++system) {
    const std::int64_t orbital_begin = layout.batch_orbital_offsets[system];
    const std::int64_t orbital_end = layout.batch_orbital_offsets[system + 1];
    const std::int64_t n = orbital_end - orbital_begin;
    const std::int64_t eigenvalue_system_begin = layout.eigenvalues.system_offsets[system];
    const std::int64_t system_matrix_begin = layout.coefficients.system_offsets[system];
    const std::int64_t matrix_elements = n * n;
    for (std::int32_t spin = 0; spin < host.spin_channels()[system]; ++spin) {
      const std::int64_t eigenvalue_begin = eigenvalue_system_begin + spin * n;
      const std::int64_t matrix_begin = system_matrix_begin + spin * matrix_elements;
      for (std::int64_t cluster_begin = 0; cluster_begin < n;) {
        std::int64_t cluster_end = cluster_begin + 1;
        while (cluster_end < n) {
          const std::int64_t previous = eigenvalue_begin + cluster_end - 1;
          const std::int64_t next = eigenvalue_begin + cluster_end;
          const double scale = std::max(
              {1.0, std::abs(expected_eigenvalues[previous]), std::abs(expected_eigenvalues[next]),
               std::abs(actual_eigenvalues[static_cast<std::size_t>(previous)]),
               std::abs(actual_eigenvalues[static_cast<std::size_t>(next)])});
          const double expected_gap =
              std::abs(expected_eigenvalues[next] - expected_eigenvalues[previous]);
          const double actual_gap =
              std::abs(actual_eigenvalues[static_cast<std::size_t>(next)] -
                       actual_eigenvalues[static_cast<std::size_t>(previous)]);
          if (std::max(expected_gap, actual_gap) > 2.0 * eigenvalue_tolerance * scale) {
            break;
          }
          ++cluster_end;
        }

        if (cluster_end == cluster_begin + 1) {
          double dot = 0.0;
          for (std::int64_t row = 0; row < n; ++row) {
            const std::int64_t index = matrix_begin + row * n + cluster_begin;
            dot += actual[static_cast<std::size_t>(index)] * expected[index];
          }
          if (dot < 0.0) {
            for (std::int64_t row = 0; row < n; ++row) {
              const std::int64_t index = matrix_begin + row * n + cluster_begin;
              actual[static_cast<std::size_t>(index)] = -actual[static_cast<std::size_t>(index)];
            }
          }
          for (std::int64_t row = 0; row < n; ++row) {
            const std::int64_t index = matrix_begin + row * n + cluster_begin;
            if (!near(actual[static_cast<std::size_t>(index)], expected[index],
                      coefficient_tolerance)) {
              std::fprintf(stderr,
                           "coefficient mismatch system=%lld spin=%d orbital=%lld row=%lld "
                           "CUDA=%.17g CPU=%.17g\n",
                           static_cast<long long>(system), static_cast<int>(spin),
                           static_cast<long long>(cluster_begin), static_cast<long long>(row),
                           actual[static_cast<std::size_t>(index)], expected[index]);
              return false;
            }
          }
        } else {
          for (std::int64_t row = 0; row < n; ++row) {
            for (std::int64_t column = 0; column < n; ++column) {
              double actual_projector = 0.0;
              double expected_projector = 0.0;
              for (std::int64_t orbital = cluster_begin; orbital < cluster_end; ++orbital) {
                actual_projector +=
                    actual[static_cast<std::size_t>(matrix_begin + row * n + orbital)] *
                    actual[static_cast<std::size_t>(matrix_begin + column * n + orbital)];
                expected_projector += expected[matrix_begin + row * n + orbital] *
                                      expected[matrix_begin + column * n + orbital];
              }
              if (!near(actual_projector, expected_projector, coefficient_tolerance)) {
                std::fprintf(stderr,
                             "coefficient projector mismatch system=%lld spin=%d "
                             "cluster=[%lld,%lld) row=%lld column=%lld CUDA=%.17g CPU=%.17g\n",
                             static_cast<long long>(system), static_cast<int>(spin),
                             static_cast<long long>(cluster_begin),
                             static_cast<long long>(cluster_end), static_cast<long long>(row),
                             static_cast<long long>(column), actual_projector, expected_projector);
                return false;
              }
            }
          }
        }
        cluster_begin = cluster_end;
      }
    }
  }
  return true;
}

bool generalized_eigensystems_match_overlap(const HostSccCase& host,
                                            const std::vector<double>& overlap,
                                            const std::vector<double>& hamiltonian,
                                            const std::vector<double>& eigenvalues,
                                            const std::vector<double>& coefficients) {
  constexpr double kTolerance = 2.0e-8;
  const auto& layout = host.wavefunction_layout();
  if (overlap.size() != static_cast<std::size_t>(layout.coefficients.element_count) ||
      hamiltonian.size() != overlap.size() ||
      eigenvalues.size() != static_cast<std::size_t>(layout.eigenvalues.element_count) ||
      coefficients.size() != overlap.size()) {
    return false;
  }
  for (std::int64_t system = 0; system < layout.batch_size; ++system) {
    const std::int64_t orbital_begin = layout.batch_orbital_offsets[system];
    const std::int64_t n =
        layout.batch_orbital_offsets[system + 1] - layout.batch_orbital_offsets[system];
    const std::int64_t matrix_begin = layout.coefficients.system_offsets[system];
    for (std::int64_t orbital = 0; orbital < n; ++orbital) {
      for (std::int64_t row = 0; row < n; ++row) {
        double hc = 0.0;
        double sc = 0.0;
        for (std::int64_t column = 0; column < n; ++column) {
          const double coefficient =
              coefficients[static_cast<std::size_t>(matrix_begin + column * n + orbital)];
          hc +=
              hamiltonian[static_cast<std::size_t>(matrix_begin + row * n + column)] * coefficient;
          sc += overlap[static_cast<std::size_t>(matrix_begin + row * n + column)] * coefficient;
        }
        if (!near(hc, sc * eigenvalues[static_cast<std::size_t>(orbital_begin + orbital)],
                  kTolerance)) {
          return false;
        }
      }
      for (std::int64_t other = 0; other < n; ++other) {
        double metric = 0.0;
        for (std::int64_t row = 0; row < n; ++row) {
          for (std::int64_t column = 0; column < n; ++column) {
            metric += coefficients[static_cast<std::size_t>(matrix_begin + row * n + orbital)] *
                      overlap[static_cast<std::size_t>(matrix_begin + row * n + column)] *
                      coefficients[static_cast<std::size_t>(matrix_begin + column * n + other)];
          }
        }
        if (!near(metric, orbital == other ? 1.0 : 0.0, kTolerance)) {
          return false;
        }
      }
    }
  }
  return true;
}

template <typename T>
bool compare_exact_values(const char* field, const std::vector<T>& actual, const T* expected,
                          std::int64_t elements) {
  if (expected == nullptr || elements < 0 || actual.size() != static_cast<std::size_t>(elements)) {
    std::fprintf(stderr, "%s has an invalid parity extent\n", field);
    return false;
  }
  for (std::int64_t index = 0; index < elements; ++index) {
    if (actual[static_cast<std::size_t>(index)] != expected[index]) {
      std::fprintf(stderr, "%s mismatch at %lld\n", field, static_cast<long long>(index));
      return false;
    }
  }
  return true;
}

/* Compare the complete persistent scalar and mixer trace after one production
 * iteration. Keeping every transfer asynchronous until the single final
 * synchronization also exercises the public descriptors as a coherent view,
 * rather than accidentally relying on host synchronization between fields. */
int compare_energy_mixer_and_scc_trace(const HostSccCase& host,
                                       const Gfn2SccIterationDeviceInput& input,
                                       const Gfn2SccIterationDeviceState& state,
                                       cudaStream_t stream, double tolerance = 3.0e-9,
                                       double history_tolerance = -1.0) {
  const double kTolerance = tolerance;
  const double kHistoryTolerance = history_tolerance < 0.0 ? tolerance : history_tolerance;
  const std::int64_t batch = host.batch_size();
  const std::int64_t mixer_vector = host.mixer_plan().total_vector_elements();
  const std::int64_t mixer_history = mixer_vector * host.mixer_plan().history_size();
  const std::int64_t mixer_omega = batch * host.mixer_plan().history_size();
  const auto& cpu_energy = host.driver_state();
  const auto& cpu_mixer = host.mixer_state();

  CHECK(state.classical_energy.es2_elements == batch);
  CHECK(state.classical_energy.es3_elements == batch);
  CHECK(state.classical_energy.aes2_elements == batch);
  CHECK(state.classical_energy.d4_two_body != nullptr);
  CHECK(state.classical_energy.d4_two_body_elements == batch);
  CHECK(state.classical_energy.explicit_point_charge != nullptr);
  CHECK(state.classical_energy.explicit_point_charge_elements == batch);
  CHECK(state.classical_energy.periodic_embedding != nullptr);
  CHECK(state.classical_energy.periodic_embedding_elements == batch);
  CHECK(state.classical_energy.classical_total_elements == batch);
  CHECK(state.free_energy.core_elements == batch);
  CHECK(state.free_energy.es2 == state.classical_energy.es2);
  CHECK(state.free_energy.es2_elements == state.classical_energy.es2_elements);
  CHECK(state.free_energy.es3 == state.classical_energy.es3);
  CHECK(state.free_energy.es3_elements == state.classical_energy.es3_elements);
  CHECK(state.free_energy.aes2 == state.classical_energy.aes2);
  CHECK(state.free_energy.aes2_elements == state.classical_energy.aes2_elements);
  CHECK(state.free_energy.d4_two_body == state.classical_energy.d4_two_body);
  CHECK(state.free_energy.d4_two_body_elements == state.classical_energy.d4_two_body_elements);
  CHECK(state.free_energy.explicit_point_charge == state.classical_energy.explicit_point_charge);
  CHECK(state.free_energy.explicit_point_charge_elements ==
        state.classical_energy.explicit_point_charge_elements);
  CHECK(state.free_energy.periodic_embedding == state.classical_energy.periodic_embedding);
  CHECK(state.free_energy.periodic_embedding_elements ==
        state.classical_energy.periodic_embedding_elements);
  CHECK(state.free_energy.entropy_elements == batch);
  CHECK(state.spin_energies != nullptr);
  CHECK(state.spin_energy_elements == batch);
  CHECK(state.free_energy.spin == state.spin_energies);
  CHECK(state.free_energy.spin_elements == state.spin_energy_elements);
  CHECK(state.free_energy.internal_energy_elements == batch);
  CHECK(state.free_energy.free_energy_elements == batch);
  CHECK(state.mixer.total_vector_elements == mixer_vector);
  CHECK(state.mixer.history_elements == mixer_history);
  CHECK(state.mixer.omega_elements == mixer_omega);
  CHECK(state.mixer.batch_elements == batch);
  CHECK(state.scc.batch_elements == batch);

  const bool d4_enabled = host.options().enable_d4;
  const bool point_charges_enabled = host.options().enable_explicit_point_charges;
  const bool periodic_enabled = host.options().enable_periodic_embedding;
  if (d4_enabled) {
    CHECK(input.classical_energy.d4_two_body != nullptr);
    CHECK(input.classical_energy.d4_two_body_elements == batch);
    CHECK(input.free_energy.d4_two_body == input.classical_energy.d4_two_body);
    CHECK(input.free_energy.d4_two_body_elements == batch);
    CHECK(cpu_energy.d4_two_body_energies != nullptr);
  } else {
    CHECK(input.classical_energy.d4_two_body == nullptr);
    CHECK(input.classical_energy.d4_two_body_elements == 0);
    CHECK(input.free_energy.d4_two_body == nullptr);
    CHECK(input.free_energy.d4_two_body_elements == 0);
    CHECK(cpu_energy.d4_two_body_energies == nullptr);
  }
  if (point_charges_enabled) {
    CHECK(input.classical_energy.explicit_point_charge != nullptr);
    CHECK(input.classical_energy.explicit_point_charge_elements == batch);
    CHECK(input.free_energy.explicit_point_charge == input.classical_energy.explicit_point_charge);
    CHECK(input.free_energy.explicit_point_charge_elements == batch);
    CHECK(cpu_energy.explicit_point_charge_energies != nullptr);
  } else {
    CHECK(input.classical_energy.explicit_point_charge == nullptr);
    CHECK(input.classical_energy.explicit_point_charge_elements == 0);
    CHECK(input.free_energy.explicit_point_charge == nullptr);
    CHECK(input.free_energy.explicit_point_charge_elements == 0);
    /* The CPU driver reserves this base component unconditionally even when
     * the geometry view disables point charges; its published value is zero. */
    CHECK(cpu_energy.explicit_point_charge_energies != nullptr);
  }
  if (periodic_enabled) {
    CHECK(input.classical_energy.periodic_embedding != nullptr);
    CHECK(input.classical_energy.periodic_embedding_elements == batch);
    CHECK(input.free_energy.periodic_embedding == input.classical_energy.periodic_embedding);
    CHECK(input.free_energy.periodic_embedding_elements == batch);
    CHECK(cpu_energy.periodic_embedding_energies != nullptr);
  } else {
    CHECK(input.classical_energy.periodic_embedding == nullptr);
    CHECK(input.classical_energy.periodic_embedding_elements == 0);
    CHECK(input.free_energy.periodic_embedding == nullptr);
    CHECK(input.free_energy.periodic_embedding_elements == 0);
    CHECK(cpu_energy.periodic_embedding_energies == nullptr);
  }

  std::vector<double> core;
  std::vector<double> es2;
  std::vector<double> es3;
  std::vector<double> aes2;
  std::vector<double> d4_two_body;
  std::vector<double> explicit_point_charge;
  std::vector<double> periodic_embedding;
  std::vector<double> classical_total;
  std::vector<double> entropy;
  std::vector<double> spin;
  std::vector<double> internal_energy;
  std::vector<double> free_energy;
  std::vector<double> current_inputs;
  std::vector<double> previous_inputs;
  std::vector<double> previous_residuals;
  std::vector<double> df_history;
  std::vector<double> u_history;
  std::vector<double> omega;
  std::vector<double> mixer_residual_rms;
  std::vector<double> mixer_residual_maximum;
  std::vector<std::uint64_t> mixer_iterations;
  std::vector<std::uint64_t> mixer_restart_counts;
  std::vector<xtbloom_status_t> mixer_statuses;
  std::vector<std::uint8_t> mixer_initialized;
  std::vector<std::uint8_t> mixer_residual_converged;
  std::vector<double> scc_free_energies;
  std::vector<double> scc_previous_free_energies;
  std::vector<double> scc_free_energy_changes;
  std::vector<double> scc_residual_rms;
  std::vector<std::uint64_t> scc_iterations;
  std::vector<xtbloom_status_t> scc_statuses;
  std::vector<std::uint8_t> scc_converged;

  CHECK(download(state.free_energy.core, batch, core, stream));
  CHECK(download(state.classical_energy.es2, batch, es2, stream));
  CHECK(download(state.classical_energy.es3, batch, es3, stream));
  CHECK(download(state.classical_energy.aes2, batch, aes2, stream));
  CHECK(download(state.classical_energy.d4_two_body, batch, d4_two_body, stream));
  CHECK(
      download(state.classical_energy.explicit_point_charge, batch, explicit_point_charge, stream));
  CHECK(download(state.classical_energy.periodic_embedding, batch, periodic_embedding, stream));
  CHECK(download(state.classical_energy.classical_total, batch, classical_total, stream));
  CHECK(download(state.free_energy.entropy, batch, entropy, stream));
  CHECK(download(state.spin_energies, batch, spin, stream));
  CHECK(download(state.free_energy.internal_energy, batch, internal_energy, stream));
  CHECK(download(state.free_energy.free_energy, batch, free_energy, stream));

  CHECK(download(state.mixer.current_inputs, mixer_vector, current_inputs, stream));
  CHECK(download(state.mixer.previous_inputs, mixer_vector, previous_inputs, stream));
  CHECK(download(state.mixer.previous_residuals, mixer_vector, previous_residuals, stream));
  CHECK(download(state.mixer.df_history, mixer_history, df_history, stream));
  CHECK(download(state.mixer.u_history, mixer_history, u_history, stream));
  CHECK(download(state.mixer.omega, mixer_omega, omega, stream));
  CHECK(download(state.mixer.residual_rms, batch, mixer_residual_rms, stream));
  CHECK(download(state.mixer.residual_maximum, batch, mixer_residual_maximum, stream));
  CHECK(download(state.mixer.iterations, batch, mixer_iterations, stream));
  CHECK(download(state.mixer.restart_counts, batch, mixer_restart_counts, stream));
  CHECK(download(state.mixer.system_statuses, batch, mixer_statuses, stream));
  CHECK(download(state.mixer.initialized, batch, mixer_initialized, stream));
  CHECK(download(state.mixer.residual_converged, batch, mixer_residual_converged, stream));

  CHECK(download(state.scc.free_energies, batch, scc_free_energies, stream));
  CHECK(download(state.scc.previous_free_energies, batch, scc_previous_free_energies, stream));
  CHECK(download(state.scc.free_energy_changes, batch, scc_free_energy_changes, stream));
  CHECK(download(state.scc.residual_rms, batch, scc_residual_rms, stream));
  CHECK(download(state.scc.iterations, batch, scc_iterations, stream));
  CHECK(download(state.scc.system_statuses, batch, scc_statuses, stream));
  CHECK(download(state.scc.converged, batch, scc_converged, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CHECK(compare_doubles("core energy", core, cpu_energy.core_energies, batch, kTolerance));
  CHECK(compare_doubles("ES2 energy", es2, cpu_energy.es2_energies, batch, kTolerance));
  CHECK(compare_doubles("ES3 energy", es3, cpu_energy.es3_energies, batch, kTolerance));
  CHECK(compare_doubles("AES2 energy", aes2, cpu_energy.aes2_energies, batch, kTolerance));
  if (d4_enabled) {
    CHECK(compare_doubles("D4 two-body energy", d4_two_body, cpu_energy.d4_two_body_energies, batch,
                          kTolerance));
  }
  CHECK(compare_doubles("explicit point-charge energy", explicit_point_charge,
                        cpu_energy.explicit_point_charge_energies, batch, kTolerance));
  if (periodic_enabled) {
    CHECK(compare_doubles("periodic embedding energy", periodic_embedding,
                          cpu_energy.periodic_embedding_energies, batch, kTolerance));
  }

  const std::vector<double> disabled_component(static_cast<std::size_t>(batch), 0.0);
  if (!d4_enabled) {
    CHECK(compare_doubles("disabled D4 two-body energy", d4_two_body, disabled_component.data(),
                          batch, kTolerance));
  }
  if (!periodic_enabled) {
    CHECK(compare_doubles("disabled periodic embedding energy", periodic_embedding,
                          disabled_component.data(), batch, kTolerance));
  }

  std::vector<double> expected_classical_total(static_cast<std::size_t>(batch));
  for (std::int64_t system = 0; system < batch; ++system) {
    double total = cpu_energy.es2_energies[system] + cpu_energy.es3_energies[system];
    total += cpu_energy.aes2_energies[system];
    if (d4_enabled) {
      total += cpu_energy.d4_two_body_energies[system];
    }
    if (point_charges_enabled) {
      total += cpu_energy.explicit_point_charge_energies[system];
    }
    if (periodic_enabled) {
      total += cpu_energy.periodic_embedding_energies[system];
    }
    expected_classical_total[static_cast<std::size_t>(system)] = total;
  }
  CHECK(compare_doubles("classical total energy", classical_total, expected_classical_total.data(),
                        batch, kTolerance));
  CHECK(compare_doubles("entropy", entropy, cpu_energy.entropies, batch, kTolerance));
  CHECK(compare_doubles("spin energy", spin, cpu_energy.spin_energies, batch, kTolerance));
  CHECK(compare_doubles("internal energy", internal_energy, cpu_energy.internal_energies, batch,
                        kTolerance));
  CHECK(compare_doubles("free energy", free_energy, cpu_energy.free_energies, batch, kTolerance));

  CHECK(compare_doubles("mixer current inputs", current_inputs, cpu_mixer.current_inputs,
                        mixer_vector, kTolerance));
  CHECK(compare_doubles("mixer previous inputs", previous_inputs, cpu_mixer.previous_inputs,
                        mixer_vector, kTolerance));
  CHECK(compare_doubles("mixer previous residuals", previous_residuals,
                        cpu_mixer.previous_residuals, mixer_vector, kTolerance));
  if (kHistoryTolerance > 0.0) {
    CHECK(compare_doubles("mixer df history", df_history, cpu_mixer.df_history, mixer_history,
                          kHistoryTolerance));
    CHECK(compare_doubles("mixer u history", u_history, cpu_mixer.u_history, mixer_history,
                          kHistoryTolerance));
    CHECK(compare_doubles("mixer omega", omega, cpu_mixer.omega, mixer_omega, kHistoryTolerance));
  }
  CHECK(compare_doubles("mixer residual RMS", mixer_residual_rms, cpu_mixer.residual_rms, batch,
                        kTolerance));
  CHECK(compare_doubles("mixer residual maximum", mixer_residual_maximum,
                        cpu_mixer.residual_maximum, batch, kTolerance));
  CHECK(compare_exact_values("mixer iterations", mixer_iterations, cpu_mixer.iterations, batch));
  CHECK(compare_exact_values("mixer restart counts", mixer_restart_counts, cpu_mixer.restart_counts,
                             batch));
  CHECK(compare_exact_values("mixer statuses", mixer_statuses, cpu_mixer.system_statuses, batch));
  CHECK(compare_exact_values("mixer initialized", mixer_initialized, cpu_mixer.initialized, batch));
  std::vector<std::uint8_t> expected_residual_converged(static_cast<std::size_t>(batch));
  for (std::int64_t system = 0; system < batch; ++system) {
    expected_residual_converged[static_cast<std::size_t>(system)] =
        cpu_mixer.residual_rms[system] < host.mixer_plan().rms_tolerance() &&
                cpu_mixer.residual_maximum[system] < host.mixer_plan().maximum_tolerance()
            ? 1u
            : 0u;
  }
  CHECK(compare_exact_values("mixer residual-converged", mixer_residual_converged,
                             expected_residual_converged.data(), batch));

  CHECK(compare_doubles("SCC free energy", scc_free_energies, cpu_energy.free_energies, batch,
                        kTolerance));
  CHECK(compare_doubles("SCC previous free energy", scc_previous_free_energies,
                        cpu_energy.previous_free_energies, batch, kTolerance));
  CHECK(compare_doubles("SCC free-energy change", scc_free_energy_changes,
                        cpu_energy.free_energy_changes, batch, kTolerance));
  CHECK(compare_doubles("SCC residual RMS", scc_residual_rms, cpu_mixer.residual_rms, batch,
                        kTolerance));
  CHECK(compare_exact_values("SCC iterations", scc_iterations, cpu_energy.iterations, batch));
  CHECK(compare_exact_values("SCC statuses", scc_statuses, cpu_energy.system_statuses, batch));
  CHECK(compare_exact_values("SCC converged", scc_converged, cpu_energy.converged, batch));
  return 0;
}

struct ProductionFixture {
  HostSccCase host;
  InputBacking backing;
  xtbloom::test::cuda::D4CommittedPairListFixture d4_pairlist;
  ProviderHandles handles;
  Gfn2SccSetupTopology topology_owner;
  Gfn2SccSetupInputs inputs_owner;
  Gfn2SccSetupEigensolver eigensolver_owner;
  Gfn2SccIterationInitializer initializer;
  DeviceAllocation topology_arena;
  DeviceAllocation input_arena;
  DeviceAllocation iteration_arena;
  DeviceAllocation eigensolver_setup_arena;
  PinnedAllocation provider_host_workspace;
  Gfn2RaggedTopologyView device_topology{};
  Gfn2WavefunctionLayoutView device_wavefunction{};
  Gfn2SccIterationDevicePlan plan_seed{};
  Gfn2SccIterationDeviceInput input_seed{};
  Gfn2SccIterationArenaRequirements arena_requirements{};
  Gfn2SccIterationDeviceState state_seed{};
  Gfn2SccIterationDeviceWorkspace workspace_seed{};
  Gfn2SccIterationReportStorage report_storage{};
  Gfn2SccSetupEigensolverBinding eigensolver_binding{};
  Gfn2SccIterationInitializationReady ready{};
  Gfn2SccIterationBinding binding{};
  bool create(bool optional_components, std::int64_t batch_size = 4, bool unrestricted_spin = false,
              bool mixed_spin_batch = false, const std::vector<SmallSystemKind>& systems = {},
              double electronic_temperature = 0.0, CouplingSelection coupling = {},
              std::uint64_t maximum_iterations = 8u, double residual_tolerance = 1.0e-10,
              double energy_tolerance = 1.0e-8, bool deterministic_debug = false) {
    if (batch_size <= 0) {
      return false;
    }
    if (unrestricted_spin && mixed_spin_batch) {
      return false;
    }
    HostSccCaseOptions options{};
    constexpr std::array<SmallSystemKind, 4> kSystems{SmallSystemKind::kH2, SmallSystemKind::kHe,
                                                      SmallSystemKind::kLiH, SmallSystemKind::kCH2};
    options.systems.clear();
    options.systems.reserve(static_cast<std::size_t>(batch_size));
    if (mixed_spin_batch) {
      options.molecular_charges.reserve(static_cast<std::size_t>(batch_size));
      options.unpaired_electrons.reserve(static_cast<std::size_t>(batch_size));
      options.spin_channels.reserve(static_cast<std::size_t>(batch_size));
      for (std::int64_t system = 0; system < batch_size; ++system) {
        /* A one-peer mixed-mode fixture must still exercise the unrestricted
         * descriptor path; larger batches alternate both spin policies. */
        const bool doublet = batch_size == 1 || system % 2 != 0;
        options.systems.push_back(
            doublet ? SmallSystemKind::kHe
                    : kSystems[static_cast<std::size_t>(system / 2) % kSystems.size()]);
        options.molecular_charges.push_back(doublet ? 1.0 : 0.0);
        options.unpaired_electrons.push_back(doublet ? 1 : 0);
        options.spin_channels.push_back(doublet ? 2 : 1);
      }
    } else {
      for (std::int64_t system = 0; system < batch_size; ++system) {
        if (!systems.empty()) {
          options.systems.push_back(systems[static_cast<std::size_t>(system) % systems.size()]);
        } else {
          options.systems.push_back(
              unrestricted_spin ? SmallSystemKind::kHe
                                : kSystems[static_cast<std::size_t>(system) % kSystems.size()]);
        }
      }
    }
    if (unrestricted_spin) {
      options.molecular_charges.assign(static_cast<std::size_t>(batch_size), 1.0);
      options.unpaired_electrons.assign(static_cast<std::size_t>(batch_size), 1);
      options.spin_channels.assign(static_cast<std::size_t>(batch_size), 2);
    }
    options.geometry_generation = kGeometryGeneration;
    options.maximum_iterations = maximum_iterations;
    options.mixer_history = 3;
    options.residual_tolerance = residual_tolerance;
    options.energy_tolerance = energy_tolerance;
    options.electronic_temperature = electronic_temperature;
    options.enable_d4 = optional_components || coupling.d4;
    options.enable_explicit_point_charges = optional_components || coupling.point_charges;
    options.enable_periodic_embedding = optional_components || coupling.periodic;

    std::string error;
    if (HostSccCase::create(options, host, error) != XTBLOOM_STATUS_SUCCESS) {
      std::fprintf(stderr, "HostSccCase::create failed: %s\n", error.c_str());
      return false;
    }
    if (!backing.prepare(host, error) || !handles.create()) {
      std::fprintf(stderr, "production SCC host/provider setup failed: %s\n", error.c_str());
      return false;
    }

    auto topology_diagnostic =
        Gfn2SccSetupTopology::create(host.basis_plan(), host.integral_plan(),
                                     host.wavefunction_layout(), kPlanToken, topology_owner);
    if (!topology_diagnostic.success() ||
        !topology_arena.allocate(topology_owner.requirements().immutable_device_bytes)) {
      std::fprintf(stderr, "production SCC topology create/allocation failed\n");
      return false;
    }
    topology_diagnostic = topology_owner.bind_device_arena_and_upload_async(
        topology_arena.get(), topology_arena.bytes(), device_topology, device_wavefunction,
        handles.stream());
    if (!topology_diagnostic.success()) {
      std::fprintf(stderr, "production SCC topology upload failed: error=%u field=%u\n",
                   static_cast<unsigned>(topology_diagnostic.error),
                   static_cast<unsigned>(topology_diagnostic.field));
      return false;
    }

    Gfn2SccSetupInputSources sources = backing.sources(host);
    /* The production setup owner seals eigensolver policy before Graph
     * construction, so deterministic-debug coverage must opt in here rather
     * than mutating the already-bound iteration plan. */
    sources.eigensolver_options.deterministic_debug = deterministic_debug;
    auto input_diagnostic = Gfn2SccSetupInputs::create(sources, topology_owner.host_topology(),
                                                       kPlanToken, inputs_owner);
    if (!input_diagnostic.success() ||
        !input_arena.allocate(inputs_owner.requirements().device_bytes)) {
      std::fprintf(stderr,
                   "production SCC immutable-input create/allocation failed: error=%u "
                   "field=%u index=%lld\n",
                   static_cast<unsigned>(input_diagnostic.error),
                   static_cast<unsigned>(input_diagnostic.field),
                   static_cast<long long>(input_diagnostic.index));
      return false;
    }
    input_diagnostic = inputs_owner.bind_device_arena_and_upload_async(
        device_topology, device_wavefunction, input_arena.get(), input_arena.bytes(), plan_seed,
        input_seed, handles.stream());
    if (!input_diagnostic.success()) {
      std::fprintf(stderr, "production SCC immutable-input upload failed: error=%u field=%u\n",
                   static_cast<unsigned>(input_diagnostic.error),
                   static_cast<unsigned>(input_diagnostic.field));
      return false;
    }
    if (host.d4_plan() != nullptr &&
        !d4_pairlist.bind(
            host.atom_offsets(), host.positions(), device_topology,
            plan_seed.d4_pairlist_cache.positions, plan_seed.d4_pairlist_cache.coordination_numbers,
            host.options().geometry_generation, plan_seed.d4_pairlist_cache, handles.stream())) {
      std::fprintf(stderr, "production SCC D4 committed pair-list setup failed\n");
      return false;
    }

    auto eigensolver_diagnostic = Gfn2SccSetupEigensolver::create(
        topology_owner, host.overlap().data(), static_cast<std::int64_t>(host.overlap().size()),
        kGeometryGeneration, kPlanToken, handles.solver(), handles.parameters(), handles.blas(),
        plan_seed.eigensolver_options, eigensolver_owner);
    if (!eigensolver_diagnostic.success()) {
      std::fprintf(stderr, "production SCC eigensolver owner create failed: error=%u field=%u\n",
                   static_cast<unsigned>(eigensolver_diagnostic.error),
                   static_cast<unsigned>(eigensolver_diagnostic.field));
      return false;
    }

    const auto& eigensolver_requirements = eigensolver_owner.requirements();
    const auto arena_diagnostic = query_gfn2_scc_iteration_arena_requirements_cuda(
        plan_seed, eigensolver_requirements.provider, arena_requirements);
    if (!arena_diagnostic.success() || !iteration_arena.allocate(arena_requirements.total_bytes) ||
        !provider_host_workspace.allocate(
            eigensolver_requirements.provider.solver_host_workspace_bytes)) {
      std::fprintf(stderr, "production SCC iteration-arena query/allocation failed: error=%u\n",
                   static_cast<unsigned>(arena_diagnostic.error));
      return false;
    }

    auto bind_arena_diagnostic = bind_gfn2_scc_iteration_arena_cuda(
        plan_seed, eigensolver_requirements.provider, arena_requirements, iteration_arena.get(),
        iteration_arena.bytes(), provider_host_workspace.get(), provider_host_workspace.bytes(),
        state_seed, workspace_seed, report_storage);
    if (!bind_arena_diagnostic.success() ||
        !eigensolver_setup_arena.allocate(eigensolver_requirements.setup_device_bytes)) {
      std::fprintf(stderr, "production SCC iteration-arena bind failed: error=%u\n",
                   static_cast<unsigned>(bind_arena_diagnostic.error));
      return false;
    }

    eigensolver_diagnostic = eigensolver_owner.bind_and_factor_overlap_async(
        device_topology, plan_seed, arena_requirements, iteration_arena.get(),
        iteration_arena.bytes(), workspace_seed, provider_host_workspace.get(),
        provider_host_workspace.bytes(), eigensolver_setup_arena.get(),
        eigensolver_setup_arena.bytes(), eigensolver_binding, handles.stream());
    if (!eigensolver_diagnostic.success()) {
      std::fprintf(stderr,
                   "production SCC overlap factorization submission failed: error=%u field=%u "
                   "index=%lld\n",
                   static_cast<unsigned>(eigensolver_diagnostic.error),
                   static_cast<unsigned>(eigensolver_diagnostic.field),
                   static_cast<long long>(eigensolver_diagnostic.index));
      return false;
    }
    plan_seed.eigensolver_batch = eigensolver_binding.batch;
    plan_seed.eigensolver_provider = eigensolver_binding.provider;
    plan_seed.overlap_cache = eigensolver_binding.cache;
    plan_seed.eigensolver_options = eigensolver_binding.options;

    const bool spin_aware_initialization = unrestricted_spin || mixed_spin_batch;
    const Gfn2SccIterationHostInitialization host_initialization =
        spin_aware_initialization
            ? fresh_initialization(topology_owner.host_wavefunction_layout(),
                                   host.wavefunction_layout(), host.wavefunction())
            : fresh_initialization(host);
    auto initialization_diagnostic = Gfn2SccIterationInitializer::create(
        plan_seed, arena_requirements, iteration_arena.get(), iteration_arena.bytes(), state_seed,
        workspace_seed, report_storage, host_initialization, initializer);
    if (!initialization_diagnostic.success()) {
      std::fprintf(stderr,
                   "production SCC initializer create failed: error=%u field=%u "
                   "index=%lld\n",
                   static_cast<unsigned>(initialization_diagnostic.error),
                   static_cast<unsigned>(initialization_diagnostic.field),
                   static_cast<long long>(initialization_diagnostic.index));
      return false;
    }
    initialization_diagnostic = initializer.upload_async(
        iteration_arena.get(), iteration_arena.bytes(), ready, handles.stream());
    if (!initialization_diagnostic.success()) {
      std::fprintf(stderr, "production SCC initializer upload failed: error=%u field=%u\n",
                   static_cast<unsigned>(initialization_diagnostic.error),
                   static_cast<unsigned>(initialization_diagnostic.field));
      return false;
    }

    const auto report_diagnostic = build_gfn2_scc_iteration_report_binding_cuda(
        report_storage, plan_seed, input_seed, state_seed, workspace_seed, binding);
    if (report_diagnostic.error != Gfn2SccIterationBindingError::kSuccess) {
      std::fprintf(stderr,
                   "production SCC report/binding build failed: error=%u field=%u "
                   "index=%lld\n",
                   static_cast<unsigned>(report_diagnostic.error),
                   static_cast<unsigned>(report_diagnostic.field),
                   static_cast<long long>(report_diagnostic.index));
      return false;
    }
    return true;
  }
};

bool finite_changed_slice(const std::vector<double>& values, std::int64_t begin, std::int64_t end,
                          double sentinel) {
  if (begin < 0 || begin > end || end > static_cast<std::int64_t>(values.size())) {
    return false;
  }
  for (std::int64_t index = begin; index < end; ++index) {
    const double value = values[static_cast<std::size_t>(index)];
    if (!std::isfinite(value) || value == sentinel) {
      return false;
    }
  }
  return true;
}

bool sentinel_slice(const std::vector<double>& values, std::int64_t begin, std::int64_t end,
                    double sentinel) {
  if (begin < 0 || begin > end || end > static_cast<std::int64_t>(values.size())) {
    return false;
  }
  for (std::int64_t index = begin; index < end; ++index) {
    if (values[static_cast<std::size_t>(index)] != sentinel) {
      return false;
    }
  }
  return true;
}

/* Full-DAG integration smoke for two He+ doublets. This is not a
 * numerical oracle: primitive-level tests own exact arithmetic parity. It
 * proves that the production setup, arena, binding, provider, transaction,
 * and publication paths agree on one real nspin=2 descriptor graph. */
int test_unrestricted_mixed_production_iteration_smoke() {
  constexpr double kSentinel = -777.125;
  ProductionFixture fixture;
  CHECK(fixture.create(false, 2, true));
  const auto& layout = fixture.host.wavefunction_layout();
  CHECK(fixture.binding.plan.wavefunction_layout.total_spin_channels == 4);
  CHECK(fixture.binding.plan.wavefunction_layout.total_spin_orbitals ==
        layout.eigenvalues.element_count);
  CHECK(fixture.binding.plan.wavefunction_layout.total_spin_matrix_elements ==
        layout.coefficients.element_count);
  CHECK(fixture.binding.plan.wavefunction_layout.total_spin_shells == layout.qsh.element_count);
  CHECK(fixture.binding.plan.wavefunction_layout.total_spin_atoms == layout.qat.element_count);
  CHECK(fixture.binding.state.occupations.occupation_elements == layout.occupations.element_count);
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  auto& state = fixture.binding.state;
  auto& workspace = fixture.binding.workspace;
  CHECK(upload_fill(workspace.hamiltonian.matrix, workspace.hamiltonian.elements, kSentinel));
  CHECK(upload_fill(state.eigenpairs.eigenvalues, state.eigenpairs.eigenvalue_elements, kSentinel));
  CHECK(
      upload_fill(state.eigenpairs.coefficients, state.eigenpairs.coefficient_elements, kSentinel));
  CHECK(
      upload_fill(state.occupations.occupations, state.occupations.occupation_elements, kSentinel));
  CHECK(upload_fill(state.density.density, state.density.density_elements, kSentinel));
  CHECK(upload_fill(state.density.energy_weighted_density, state.density.weighted_density_elements,
                    kSentinel));
  CHECK(upload_fill(state.density.channel_band_energies, state.density.channel_band_energy_elements,
                    kSentinel));
  CHECK(upload_fill(state.density.channel_occupation_sums,
                    state.density.channel_occupation_sum_elements, kSentinel));
  CHECK(upload_fill(state.raw_population.qsh, state.raw_population.qsh_elements, kSentinel));
  CHECK(upload_fill(state.raw_population.qat, state.raw_population.qat_elements, kSentinel));
  CHECK(upload_fill(state.raw_population.dipole, state.raw_population.dipole_elements, kSentinel));
  CHECK(upload_fill(state.raw_population.quadrupole, state.raw_population.quadrupole_elements,
                    kSentinel));
  CHECK(upload_fill(state.spin_energies, state.spin_energy_elements, kSentinel));
  CHECK(upload_fill(state.free_energy.core, state.free_energy.core_elements, kSentinel));
  CHECK(upload_fill(state.free_energy.entropy, state.free_energy.entropy_elements, kSentinel));
  CHECK(upload_fill(state.free_energy.internal_energy, state.free_energy.internal_energy_elements,
                    kSentinel));
  CHECK(upload_fill(state.free_energy.free_energy, state.free_energy.free_energy_elements,
                    kSentinel));
  CHECK(upload_fill(state.scc.free_energies, state.scc.batch_elements, kSentinel));
  CHECK(upload_fill(state.scc.previous_free_energies, state.scc.batch_elements, kSentinel));
  CHECK(upload_fill(state.scc.free_energy_changes, state.scc.batch_elements, kSentinel));
  CHECK(upload_fill(state.scc.residual_rms, state.scc.batch_elements, kSentinel));

  const std::array<std::uint64_t, 2> iterations{{0u, 0u}};
  const std::array<xtbloom_status_t, 2> statuses{{XTBLOOM_STATUS_SUCCESS, XTBLOOM_STATUS_SUCCESS}};
  const std::array<std::uint8_t, 2> converged{{0u, 1u}};
  CUDA_CHECK(cudaMemcpy(state.scc.iterations, iterations.data(), sizeof(iterations),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(state.scc.system_statuses, statuses.data(), sizeof(statuses),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(state.scc.converged, converged.data(), sizeof(converged), cudaMemcpyHostToDevice));

  const auto binding_diagnostic = validate_gfn2_scc_iteration_binding_cuda(
      fixture.binding.plan, fixture.binding.input, state, workspace);
  CHECK(binding_diagnostic.error == Gfn2SccIterationBindingError::kSuccess);
  const Gfn2SccIterationLaunchResult launch =
      launch_gfn2_scc_iteration_cuda(fixture.binding, fixture.handles.stream());
  if (!launch.success()) {
    std::fprintf(stderr,
                 "unrestricted production launch failed: status=%u stage=%u binding_error=%u "
                 "binding_field=%u cuda=%d cublas=%d cusolver=%d\n",
                 static_cast<unsigned>(launch.status), static_cast<unsigned>(launch.stage),
                 static_cast<unsigned>(launch.binding.error),
                 static_cast<unsigned>(launch.binding.field), static_cast<int>(launch.cuda_status),
                 static_cast<int>(launch.cublas_status), static_cast<int>(launch.cusolver_status));
  }
  CHECK(launch.success());

  std::vector<double> hamiltonian;
  std::vector<double> eigenvalues;
  std::vector<double> coefficients;
  std::vector<double> occupations;
  std::vector<double> density;
  std::vector<double> weighted_density;
  std::vector<double> channel_band_energies;
  std::vector<double> channel_occupation_sums;
  std::vector<double> qsh;
  std::vector<double> staged_raw_qsh;
  std::vector<double> qat;
  std::vector<double> dipole;
  std::vector<double> quadrupole;
  std::vector<double> spin_energies;
  std::vector<double> free_spin;
  std::vector<double> free_energies;
  std::vector<double> scc_free_energies;
  std::vector<std::uint64_t> scc_iterations;
  std::vector<std::uint8_t> active_mask;
  CHECK(download(workspace.hamiltonian.matrix, workspace.hamiltonian.elements, hamiltonian,
                 fixture.handles.stream()));
  CHECK(download(state.eigenpairs.eigenvalues, state.eigenpairs.eigenvalue_elements, eigenvalues,
                 fixture.handles.stream()));
  CHECK(download(state.eigenpairs.coefficients, state.eigenpairs.coefficient_elements, coefficients,
                 fixture.handles.stream()));
  CHECK(download(state.occupations.occupations, state.occupations.occupation_elements, occupations,
                 fixture.handles.stream()));
  CHECK(download(state.density.density, state.density.density_elements, density,
                 fixture.handles.stream()));
  CHECK(download(state.density.energy_weighted_density, state.density.weighted_density_elements,
                 weighted_density, fixture.handles.stream()));
  CHECK(download(state.density.channel_band_energies, state.density.channel_band_energy_elements,
                 channel_band_energies, fixture.handles.stream()));
  CHECK(download(state.density.channel_occupation_sums,
                 state.density.channel_occupation_sum_elements, channel_occupation_sums,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.qsh, state.raw_population.qsh_elements, qsh,
                 fixture.handles.stream()));
  CHECK(download(workspace.staged_raw_population.qsh, workspace.staged_raw_population.qsh_elements,
                 staged_raw_qsh, fixture.handles.stream()));
  CHECK(download(state.raw_population.qat, state.raw_population.qat_elements, qat,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.dipole, state.raw_population.dipole_elements, dipole,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.quadrupole, state.raw_population.quadrupole_elements,
                 quadrupole, fixture.handles.stream()));
  CHECK(download(state.spin_energies, state.spin_energy_elements, spin_energies,
                 fixture.handles.stream()));
  CHECK(download(state.free_energy.spin, state.free_energy.spin_elements, free_spin,
                 fixture.handles.stream()));
  CHECK(download(state.free_energy.free_energy, state.free_energy.free_energy_elements,
                 free_energies, fixture.handles.stream()));
  CHECK(download(state.scc.free_energies, state.scc.batch_elements, scc_free_energies,
                 fixture.handles.stream()));
  CHECK(download(state.scc.iterations, state.scc.batch_elements, scc_iterations,
                 fixture.handles.stream()));
  CHECK(download(workspace.ledger.active_mask, workspace.ledger.batch_elements, active_mask,
                 fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  const auto active_slice_changed = [&](const std::vector<double>& values, const auto& field) {
    return finite_changed_slice(values, field.system_offsets[0], field.system_offsets[1],
                                kSentinel);
  };
  const auto inactive_slice_is_sentinel = [&](const std::vector<double>& values,
                                              const auto& field) {
    return sentinel_slice(values, field.system_offsets[1], field.system_offsets[2], kSentinel);
  };

  CHECK(active_slice_changed(hamiltonian, layout.coefficients));
  CHECK(inactive_slice_is_sentinel(hamiltonian, layout.coefficients));
  CHECK(active_slice_changed(eigenvalues, layout.eigenvalues));
  CHECK(inactive_slice_is_sentinel(eigenvalues, layout.eigenvalues));
  CHECK(active_slice_changed(coefficients, layout.coefficients));
  CHECK(inactive_slice_is_sentinel(coefficients, layout.coefficients));
  CHECK(active_slice_changed(occupations, layout.occupations));
  CHECK(inactive_slice_is_sentinel(occupations, layout.occupations));
  const std::int64_t active_orbitals =
      layout.batch_orbital_offsets[1] - layout.batch_orbital_offsets[0];
  CHECK(occupations[static_cast<std::size_t>(layout.occupations.system_offsets[0])] > 0.5);
  CHECK(occupations[static_cast<std::size_t>(layout.occupations.system_offsets[0] +
                                             active_orbitals)] < 0.5);
  CHECK(active_slice_changed(density, layout.density));
  CHECK(inactive_slice_is_sentinel(density, layout.density));
  CHECK(active_slice_changed(weighted_density, layout.energy_weighted_density));
  CHECK(inactive_slice_is_sentinel(weighted_density, layout.energy_weighted_density));
  CHECK(active_slice_changed(qsh, layout.qsh));
  CHECK(inactive_slice_is_sentinel(qsh, layout.qsh));
  CHECK(active_slice_changed(qat, layout.qat));
  CHECK(inactive_slice_is_sentinel(qat, layout.qat));
  CHECK(active_slice_changed(dipole, layout.dipole));
  CHECK(inactive_slice_is_sentinel(dipole, layout.dipole));
  CHECK(active_slice_changed(quadrupole, layout.quadrupole));
  CHECK(inactive_slice_is_sentinel(quadrupole, layout.quadrupole));
  const std::int64_t active_shells = layout.batch_shell_offsets[1] - layout.batch_shell_offsets[0];
  const std::int64_t magnetization_begin = layout.qsh.system_offsets[0] + active_shells;
  const std::int64_t magnetization_end = layout.qsh.system_offsets[1];
  double magnetization_sum = 0.0;
  for (std::int64_t shell = magnetization_begin; shell < magnetization_end; ++shell) {
    CHECK(std::isfinite(staged_raw_qsh[static_cast<std::size_t>(shell)]));
    magnetization_sum += staged_raw_qsh[static_cast<std::size_t>(shell)];
  }
  /* The public qsh is the damped next mixed state until convergence. The
   * staged Mulliken population retains the raw one-electron spin invariant. */
  CHECK(std::abs(std::abs(magnetization_sum) - 1.0) < 1.0e-10);
  const auto& spin_layout = fixture.topology_owner.host_wavefunction_layout();
  CHECK(finite_changed_slice(channel_band_energies, spin_layout.spin_channel_offsets[0],
                             spin_layout.spin_channel_offsets[1], kSentinel));
  CHECK(sentinel_slice(channel_band_energies, spin_layout.spin_channel_offsets[1],
                       spin_layout.spin_channel_offsets[2], kSentinel));
  CHECK(finite_changed_slice(channel_occupation_sums, spin_layout.spin_channel_offsets[0],
                             spin_layout.spin_channel_offsets[1], kSentinel));
  CHECK(sentinel_slice(channel_occupation_sums, spin_layout.spin_channel_offsets[1],
                       spin_layout.spin_channel_offsets[2], kSentinel));
  CHECK(std::isfinite(spin_energies[0]) && spin_energies[0] != kSentinel);
  CHECK(spin_energies[1] == kSentinel);
  CHECK(free_spin == spin_energies);
  CHECK(std::isfinite(free_energies[0]) && free_energies[0] != kSentinel);
  CHECK(free_energies[1] == kSentinel);
  CHECK(std::isfinite(scc_free_energies[0]) && scc_free_energies[0] != kSentinel);
  CHECK(scc_free_energies[1] == kSentinel);
  CHECK(scc_iterations[0] == 1u);
  CHECK(scc_iterations[1] == 0u);
  CHECK(active_mask == std::vector<std::uint8_t>({1u, 0u}));
  return 0;
}

int test_production_iteration_cpu_parity(bool optional_components, std::int64_t batch_size = 4,
                                         bool unrestricted_spin = false, bool one_step_only = false,
                                         bool mixed_spin_batch = false,
                                         const std::vector<SmallSystemKind>& systems = {},
                                         double electronic_temperature = 0.0,
                                         CouplingSelection coupling = {}) {
  ProductionFixture fixture;
  CHECK(fixture.create(optional_components, batch_size, unrestricted_spin, mixed_spin_batch,
                       systems, electronic_temperature, coupling));

  /* The first transition is only comparable when fresh initialization packs
   * every ragged system into the same mixer vector used by the CPU oracle. */
  std::vector<double> initial_mixer_inputs;
  CHECK(download(fixture.binding.state.mixer.current_inputs,
                 fixture.binding.state.mixer.total_vector_elements, initial_mixer_inputs,
                 fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  CHECK(compare_doubles("initial mixer current inputs", initial_mixer_inputs,
                        fixture.host.mixer_state().current_inputs,
                        fixture.host.mixer_plan().total_vector_elements(), 0.0));

  const Gfn2SccIterationLaunchResult launch =
      launch_gfn2_scc_iteration_cuda(fixture.binding, fixture.handles.stream());
  if (!launch.success()) {
    std::fprintf(stderr,
                 "production SCC launch failed: status=%u stage=%u binding_error=%u "
                 "binding_field=%u cuda=%d cublas=%d cusolver=%d\n",
                 static_cast<unsigned>(launch.status), static_cast<unsigned>(launch.stage),
                 static_cast<unsigned>(launch.binding.error),
                 static_cast<unsigned>(launch.binding.field), static_cast<int>(launch.cuda_status),
                 static_cast<int>(launch.cublas_status), static_cast<int>(launch.cusolver_status));
  }
  CHECK(launch.success());

  std::string error;
  const xtbloom_status_t cpu_status = fixture.host.run_one_iteration(error);
  if (cpu_status != XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "CPU production SCC iteration failed: status=%d error=%s\n", cpu_status,
                 error.c_str());
  }
  CHECK(cpu_status == XTBLOOM_STATUS_SUCCESS);

  const auto& layout = fixture.host.wavefunction_layout();
  const auto& state = fixture.binding.state;
  const auto& workspace = fixture.binding.workspace;
  std::vector<double> eigenvalues;
  std::vector<double> coefficients;
  std::vector<double> occupations;
  std::vector<double> density;
  std::vector<double> weighted_density;
  std::vector<double> public_qsh;
  std::vector<double> public_qat;
  std::vector<double> public_dipoles;
  std::vector<double> public_quadrupoles;
  std::vector<double> raw_qsh;
  std::vector<double> raw_qat;
  std::vector<double> raw_dipoles;
  std::vector<double> raw_quadrupoles;
  std::vector<double> free_energy;
  std::vector<double> current_inputs;
  std::vector<std::uint64_t> iterations;
  std::vector<xtbloom_status_t> statuses;
  std::vector<std::uint8_t> converged;

  CHECK(download(state.eigenpairs.eigenvalues, state.eigenpairs.eigenvalue_elements, eigenvalues,
                 fixture.handles.stream()));
  CHECK(download(state.eigenpairs.coefficients, state.eigenpairs.coefficient_elements, coefficients,
                 fixture.handles.stream()));
  CHECK(download(state.occupations.occupations, state.occupations.occupation_elements, occupations,
                 fixture.handles.stream()));
  CHECK(download(state.density.density, state.density.density_elements, density,
                 fixture.handles.stream()));
  CHECK(download(state.density.energy_weighted_density, state.density.weighted_density_elements,
                 weighted_density, fixture.handles.stream()));
  CHECK(download(state.raw_population.qsh, state.raw_population.qsh_elements, public_qsh,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.qat, state.raw_population.qat_elements, public_qat,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.dipole, state.raw_population.dipole_elements, public_dipoles,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.quadrupole, state.raw_population.quadrupole_elements,
                 public_quadrupoles, fixture.handles.stream()));
  CHECK(download(workspace.staged_raw_population.qsh, workspace.staged_raw_population.qsh_elements,
                 raw_qsh, fixture.handles.stream()));
  CHECK(download(workspace.staged_raw_population.qat, workspace.staged_raw_population.qat_elements,
                 raw_qat, fixture.handles.stream()));
  CHECK(download(workspace.staged_raw_population.dipole,
                 workspace.staged_raw_population.dipole_elements, raw_dipoles,
                 fixture.handles.stream()));
  CHECK(download(workspace.staged_raw_population.quadrupole,
                 workspace.staged_raw_population.quadrupole_elements, raw_quadrupoles,
                 fixture.handles.stream()));
  CHECK(download(state.free_energy.free_energy, state.free_energy.free_energy_elements, free_energy,
                 fixture.handles.stream()));
  CHECK(download(state.mixer.current_inputs, state.mixer.total_vector_elements, current_inputs,
                 fixture.handles.stream()));
  CHECK(download(state.scc.iterations, state.scc.batch_elements, iterations,
                 fixture.handles.stream()));
  CHECK(download(state.scc.system_statuses, state.scc.batch_elements, statuses,
                 fixture.handles.stream()));
  CHECK(
      download(state.scc.converged, state.scc.batch_elements, converged, fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  for (std::int64_t system = 0; system < fixture.host.batch_size(); ++system) {
    CHECK(iterations[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().iterations[system]);
    CHECK(statuses[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().system_statuses[system]);
    CHECK(converged[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().converged[system]);
  }

  CHECK(compare_doubles("eigenvalues", eigenvalues, fixture.host.wavefunction().eigenvalues,
                        layout.eigenvalues.element_count, 3.0e-9));
  CHECK(compare_coefficients(fixture.host, eigenvalues, coefficients, 3.0e-9, 3.0e-8));
  CHECK(compare_doubles("occupations", occupations, fixture.host.wavefunction().occupations,
                        layout.occupations.element_count, 3.0e-10));
  CHECK(compare_doubles("density", density, fixture.host.wavefunction().density,
                        layout.density.element_count, 3.0e-9));
  CHECK(compare_doubles("energy-weighted density", weighted_density,
                        fixture.host.wavefunction().energy_weighted_density,
                        layout.energy_weighted_density.element_count, 3.0e-9));
  for (std::int64_t system = 0; system < layout.batch_size; ++system) {
    for (std::int64_t index = layout.qsh.system_offsets[system];
         index < layout.qsh.system_offsets[system + 1]; ++index) {
      if (!near(public_qsh[static_cast<std::size_t>(index)], fixture.host.wavefunction().qsh[index],
                3.0e-9)) {
        std::fprintf(stderr,
                     "public qsh system=%lld nspin=%d range=[%lld,%lld) converged=%u "
                     "cuda=%.17g cpu=%.17g raw_cuda=%.17g raw_cpu=%.17g\n",
                     static_cast<long long>(system), fixture.host.spin_channels()[system],
                     static_cast<long long>(layout.qsh.system_offsets[system]),
                     static_cast<long long>(layout.qsh.system_offsets[system + 1]),
                     static_cast<unsigned>(converged[static_cast<std::size_t>(system)]),
                     public_qsh[static_cast<std::size_t>(index)],
                     fixture.host.wavefunction().qsh[index],
                     raw_qsh[static_cast<std::size_t>(index)],
                     fixture.host.driver_workspace().raw_qsh[index]);
      }
    }
  }
  CHECK(compare_doubles("public qsh", public_qsh, fixture.host.wavefunction().qsh,
                        layout.qsh.element_count, 3.0e-9));
  CHECK(compare_doubles("public qat", public_qat, fixture.host.wavefunction().qat,
                        layout.qat.element_count, 3.0e-9));
  CHECK(compare_doubles("public dipoles", public_dipoles, fixture.host.wavefunction().dipole,
                        layout.dipole.element_count, 3.0e-9));
  CHECK(compare_doubles("public quadrupoles", public_quadrupoles,
                        fixture.host.wavefunction().quadrupole, layout.quadrupole.element_count,
                        3.0e-9));

  const auto& cpu_workspace = fixture.host.driver_workspace();
  CHECK(
      compare_doubles("raw qsh", raw_qsh, cpu_workspace.raw_qsh, layout.qsh.element_count, 3.0e-9));
  CHECK(
      compare_doubles("raw qat", raw_qat, cpu_workspace.raw_qat, layout.qat.element_count, 3.0e-9));
  CHECK(compare_doubles("raw dipoles", raw_dipoles, cpu_workspace.raw_dipoles,
                        layout.dipole.element_count, 3.0e-9));
  CHECK(compare_doubles("raw quadrupoles", raw_quadrupoles, cpu_workspace.raw_quadrupoles,
                        layout.quadrupole.element_count, 3.0e-9));
  CHECK(compare_doubles("free energy", free_energy, fixture.host.driver_state().free_energies,
                        fixture.host.batch_size(), 3.0e-9));
  CHECK(compare_doubles("mixer current inputs", current_inputs,
                        fixture.host.mixer_state().current_inputs,
                        fixture.host.mixer_plan().total_vector_elements(), 3.0e-9));

  CHECK(compare_energy_mixer_and_scc_trace(fixture.host, fixture.binding.input, state,
                                           fixture.handles.stream()) == 0);

  if (one_step_only) {
    return 0;
  }

  /* Reuse the exact same production binding for a second iteration. This is
   * the steady-state contract consumed by Graph replay and proves that no
   * setup owner or descriptor must be rebuilt between SCC launches. */
  const Gfn2SccIterationLaunchResult repeat_launch =
      launch_gfn2_scc_iteration_cuda(fixture.binding, fixture.handles.stream());
  CHECK(repeat_launch.success());
  CHECK(fixture.host.run_one_iteration(error) == XTBLOOM_STATUS_SUCCESS);

  CHECK(download(state.density.density, state.density.density_elements, density,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.qsh, state.raw_population.qsh_elements, public_qsh,
                 fixture.handles.stream()));
  CHECK(download(state.free_energy.free_energy, state.free_energy.free_energy_elements, free_energy,
                 fixture.handles.stream()));
  CHECK(download(state.mixer.current_inputs, state.mixer.total_vector_elements, current_inputs,
                 fixture.handles.stream()));
  CHECK(download(state.scc.iterations, state.scc.batch_elements, iterations,
                 fixture.handles.stream()));
  CHECK(download(state.scc.system_statuses, state.scc.batch_elements, statuses,
                 fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  CHECK(compare_doubles("repeat density", density, fixture.host.wavefunction().density,
                        layout.density.element_count, 3.0e-9));
  CHECK(compare_doubles("repeat public qsh", public_qsh, fixture.host.wavefunction().qsh,
                        layout.qsh.element_count, 3.0e-9));
  CHECK(compare_doubles("repeat free energy", free_energy,
                        fixture.host.driver_state().free_energies, fixture.host.batch_size(),
                        3.0e-9));
  CHECK(compare_doubles("repeat mixer current inputs", current_inputs,
                        fixture.host.mixer_state().current_inputs,
                        fixture.host.mixer_plan().total_vector_elements(), 3.0e-9));
  for (std::int64_t system = 0; system < fixture.host.batch_size(); ++system) {
    CHECK(iterations[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().iterations[system]);
    CHECK(statuses[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().system_statuses[system]);
  }
  CHECK(compare_energy_mixer_and_scc_trace(fixture.host, fixture.binding.input, state,
                                           fixture.handles.stream()) == 0);
  return 0;
}

int test_mixed_spin_batch_one_step_cpu_parity() {
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    const int status = test_production_iteration_cpu_parity(false, batch_size, false, true, true);
    if (status != 0) {
      std::fprintf(stderr, "mixed-spin one-step CPU parity failed for B=%lld\n",
                   static_cast<long long>(batch_size));
      return status;
    }
  }
  return 0;
}

/* Finite-temperature system set that always places a near-degenerate stretched
 * H2 first so the composed iteration exercises the difficult frontier Fermi
 * path while the remaining peers provide ordinary finite-T fractional filling. */
const std::vector<SmallSystemKind>& finite_temperature_systems() {
  static const std::vector<SmallSystemKind> systems{SmallSystemKind::kH2Stretched,
                                                    SmallSystemKind::kHe, SmallSystemKind::kLiH,
                                                    SmallSystemKind::kCH2};
  return systems;
}

/* The stretched-H2 peer (always system 0) must publish genuinely fractional
 * frontier occupations and keep its HOMO/LUMO gap below the electronic
 * temperature. This proves the parity case is actually exercising the
 * near-degenerate/difficult occupation regime rather than a trivial
 * T -> 0 fill, and that CPU/CUDA agree on both the splitting and the fill. */
int verify_near_degenerate_fractional_frontier(const HostSccCase& host) {
  constexpr std::int64_t kStretchedSystem = 0;
  const auto& layout = host.wavefunction_layout();
  if (host.batch_size() <= kStretchedSystem) {
    return __LINE__;
  }
  const auto& eigenvalues = host.wavefunction().eigenvalues;
  const std::int64_t n = layout.batch_orbital_offsets[kStretchedSystem + 1] -
                         layout.batch_orbital_offsets[kStretchedSystem];
  if (n < 2 || layout.batch_orbital_offsets[kStretchedSystem] != 0) {
    return __LINE__;
  }
  /* Restricted closed-shell stretched H2: the sigma_g/sigma_u* pair is the
   * sole frontier and must be genuinely near-degenerate (exact binary64
   * degeneracy is acceptable and even stronger). */
  const double homu_gap = eigenvalues[1] - eigenvalues[0];
  if (!(homu_gap >= 0.0) || !(homu_gap < 5.0e-3)) {
    std::fprintf(stderr, "stretched-H2 frontier gap %.6e is not near-degenerate\n", homu_gap);
    return __LINE__;
  }
  const double temperature = host.options().electronic_temperature;
  if (!(homu_gap < temperature)) {
    std::fprintf(stderr, "stretched-H2 gap %.6e exceeds kT %.6e\n", homu_gap, temperature);
    return __LINE__;
  }
  const auto& occupations = host.wavefunction().occupations;
  bool fractional = false;
  for (std::int64_t orbital = 0; orbital < n; ++orbital) {
    const double value = occupations[static_cast<std::size_t>(orbital)];
    if (value > 1.0e-3 && value < 1.0 - 1.0e-3) {
      fractional = true;
      break;
    }
  }
  if (!fractional) {
    std::fprintf(stderr, "stretched-H2 frontier occupations are not fractional at kT=%.6e\n",
                 temperature);
    for (std::int64_t orbital = 0; orbital < n; ++orbital) {
      std::fprintf(stderr, "  occ[%lld]=%.17g\n", static_cast<long long>(orbital),
                   occupations[static_cast<std::size_t>(orbital)]);
    }
    return __LINE__;
  }
  return 0;
}

int test_production_iteration_finite_temperature_cpu_parity() {
  const double kDefaultTemperature = XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE;
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    const int status = test_production_iteration_cpu_parity(
        false, batch_size, false, true, false, finite_temperature_systems(), kDefaultTemperature);
    if (status != 0) {
      std::fprintf(stderr, "finite-temperature one-step CPU parity failed for B=%lld\n",
                   static_cast<long long>(batch_size));
      return status;
    }
    ProductionFixture fixture;
    CHECK(fixture.create(false, batch_size, false, false, finite_temperature_systems(),
                         kDefaultTemperature));
    std::string error;
    CHECK(fixture.host.run_one_iteration(error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(verify_near_degenerate_fractional_frontier(fixture.host) == 0);
  }
  /* A hotter finite-temperature regime spreads fractional occupations over
   * every peer while the stretched-H2 frontier stays genuinely difficult. */
  constexpr double kHotTemperature = 2.0e-2;
  {
    const int status = test_production_iteration_cpu_parity(
        false, 8, false, true, false, finite_temperature_systems(), kHotTemperature);
    if (status != 0) {
      std::fprintf(stderr, "hot-temperature one-step CPU parity failed\n");
      return status;
    }
    ProductionFixture fixture;
    CHECK(fixture.create(false, 8, false, false, finite_temperature_systems(), kHotTemperature));
    std::string error;
    CHECK(fixture.host.run_one_iteration(error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(verify_near_degenerate_fractional_frontier(fixture.host) == 0);
  }
  /* Optional couplings at finite temperature compose all charge-dependent
   * environments with a fractional density in one parity run. */
  {
    const int status = test_production_iteration_cpu_parity(
        true, 8, false, true, false, finite_temperature_systems(), kDefaultTemperature);
    if (status != 0) {
      std::fprintf(stderr, "finite-temperature optional-coupling one-step parity failed\n");
      return status;
    }
  }
  return 0;
}

int test_individual_coupling_cpu_parity() {
  struct CouplingCase {
    const char* name;
    CouplingSelection selection;
  };
  constexpr CouplingCase cases[]{
      {"D4", {true, false, false}},
      {"explicit point charge", {false, true, false}},
      {"periodic", {false, false, true}},
      {"combined", {true, true, true}},
  };
  for (const CouplingCase& coupling : cases) {
    const int status = test_production_iteration_cpu_parity(false, 8, false, true, false, {}, 0.0,
                                                            coupling.selection);
    if (status != 0) {
      std::fprintf(stderr, "%s one-step CPU parity failed\n", coupling.name);
      return status;
    }
  }
  return 0;
}

/* Defined later in this translation unit; declared here for the finite-
 * temperature loop parity cover below. Default arguments live on the
 * definition only. */
int test_production_loop_cpu_parity(std::int64_t batch_size, bool optional_components,
                                    bool use_default_stream, bool use_ordered_stream,
                                    std::uint64_t resumed_iterations, bool mixed_spin_batch,
                                    const std::vector<SmallSystemKind>& systems,
                                    double electronic_temperature);

int test_production_loop_finite_temperature_cpu_parity() {
  const double kDefaultTemperature = XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE;
  const int status = test_production_loop_cpu_parity(
      8, false, false, false, 0u, false, finite_temperature_systems(), kDefaultTemperature);
  if (status != 0) {
    std::fprintf(stderr, "finite-temperature full-loop CPU parity failed\n");
    return status;
  }
  return 0;
}

int test_changed_device_overlap_is_consumed_by_production_scc() {
  constexpr std::uint64_t kChangedGeneration = kGeometryGeneration + 1u;
  ProductionFixture fixture;
  CHECK(fixture.create(false));

  std::vector<Gfn2SccCacheProvenanceBinding> provenance;
  CHECK(download(fixture.binding.plan.provenance.cache_bindings,
                 fixture.binding.plan.provenance.cache_binding_count, provenance,
                 fixture.handles.stream()));
  std::vector<double> before_factors;
  CHECK(download(fixture.eigensolver_binding.cache.cholesky_factors,
                 fixture.eigensolver_binding.cache.factor_elements, before_factors,
                 fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  std::vector<double> changed_overlap = fixture.host.overlap();
  const auto& layout = fixture.host.wavefunction_layout();
  for (std::int64_t system = 0; system < layout.batch_size; ++system) {
    const std::int64_t n =
        layout.batch_orbital_offsets[system + 1] - layout.batch_orbital_offsets[system];
    const std::int64_t matrix_begin = layout.coefficients.system_offsets[system];
    const double shift = 0.0075 * static_cast<double>(system + 1);
    for (std::int64_t orbital = 0; orbital < n; ++orbital) {
      changed_overlap[static_cast<std::size_t>(matrix_begin + orbital * n + orbital)] += shift;
    }
  }

  DeviceAllocation device_overlap;
  CHECK(device_overlap.allocate(changed_overlap.size() * sizeof(double)));
  CUDA_CHECK(cudaMemcpyAsync(device_overlap.get(), changed_overlap.data(),
                             changed_overlap.size() * sizeof(double), cudaMemcpyHostToDevice,
                             fixture.handles.stream()));
  CUDA_CHECK(cudaMemcpyAsync(const_cast<double*>(fixture.binding.input.hamiltonian.overlap),
                             changed_overlap.data(), changed_overlap.size() * sizeof(double),
                             cudaMemcpyHostToDevice, fixture.handles.stream()));

  std::vector<std::uint64_t> generations(static_cast<std::size_t>(fixture.host.batch_size()),
                                         kChangedGeneration);
  CUDA_CHECK(cudaMemcpyAsync(fixture.binding.plan.geometry_cache.geometry_generations,
                             generations.data(), generations.size() * sizeof(std::uint64_t),
                             cudaMemcpyHostToDevice, fixture.handles.stream()));
  for (auto& record : provenance) {
    if (record.provenance.generation_scope == Gfn2GenerationScope::kBatch) {
      record.provenance.geometry_generation = kChangedGeneration;
    }
  }
  CUDA_CHECK(cudaMemcpyAsync(
      const_cast<Gfn2SccCacheProvenanceBinding*>(fixture.binding.plan.provenance.cache_bindings),
      provenance.data(), provenance.size() * sizeof(Gfn2SccCacheProvenanceBinding),
      cudaMemcpyHostToDevice, fixture.handles.stream()));

  const auto refactor = fixture.eigensolver_owner.refactor_overlap_from_device_async(
      fixture.eigensolver_setup_arena.get(), fixture.eigensolver_setup_arena.bytes(),
      fixture.eigensolver_binding, static_cast<const double*>(device_overlap.get()),
      static_cast<std::int64_t>(changed_overlap.size()), kChangedGeneration,
      fixture.handles.stream());
  CHECK(refactor.success());

  Gfn2SccIterationDevicePlan changed_plan = fixture.binding.plan;
  changed_plan.geometry_generation = kChangedGeneration;
  changed_plan.provenance.expected_geometry_generation = kChangedGeneration;
  changed_plan.es2_cache.geometry_generation = kChangedGeneration;
  changed_plan.aes2_cache.geometry_generation = kChangedGeneration;
  Gfn2SccIterationBinding changed_binding{};
  CHECK(bind_gfn2_scc_iteration_cuda(changed_plan, fixture.binding.input, fixture.binding.state,
                                     fixture.binding.workspace, changed_binding)
            .error == Gfn2SccIterationBindingError::kSuccess);

  const Gfn2SccIterationLaunchResult launch =
      launch_gfn2_restricted_scc_iteration_cuda(changed_binding, fixture.handles.stream());
  CHECK(launch.success());

  std::vector<double> after_factors;
  std::vector<std::uint64_t> factor_generations;
  std::vector<double> hamiltonian;
  std::vector<double> eigenvalues;
  std::vector<double> coefficients;
  CHECK(download(fixture.eigensolver_binding.cache.cholesky_factors,
                 fixture.eigensolver_binding.cache.factor_elements, after_factors,
                 fixture.handles.stream()));
  CHECK(download(fixture.eigensolver_binding.cache.geometry_generations,
                 fixture.eigensolver_binding.cache.generation_elements, factor_generations,
                 fixture.handles.stream()));
  CHECK(download(changed_binding.workspace.hamiltonian.matrix,
                 changed_binding.workspace.hamiltonian.elements, hamiltonian,
                 fixture.handles.stream()));
  CHECK(download(changed_binding.state.eigenpairs.eigenvalues,
                 changed_binding.state.eigenpairs.eigenvalue_elements, eigenvalues,
                 fixture.handles.stream()));
  CHECK(download(changed_binding.state.eigenpairs.coefficients,
                 changed_binding.state.eigenpairs.coefficient_elements, coefficients,
                 fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  CHECK(after_factors != before_factors);
  CHECK(std::all_of(factor_generations.begin(), factor_generations.end(),
                    [](std::uint64_t value) { return value == kChangedGeneration; }));
  CHECK(generalized_eigensystems_match_overlap(fixture.host, changed_overlap, hamiltonian,
                                               eigenvalues, coefficients));
  return 0;
}

int run_host_fixed_scc_loop(HostSccCase& host) {
  std::string error;
  for (std::uint64_t iteration = 0u; iteration < host.options().maximum_iterations; ++iteration) {
    const xtbloom_status_t status = host.run_one_iteration(error);
    if (status != XTBLOOM_STATUS_SUCCESS && status != XTBLOOM_STATUS_SCC_NOT_CONVERGED &&
        status != XTBLOOM_STATUS_EIGENSOLVER_FAILED) {
      std::fprintf(stderr, "CPU Graph reference failed at %llu: status=%d error=%s\n",
                   static_cast<unsigned long long>(iteration), status, error.c_str());
      return __LINE__;
    }
  }
  return 0;
}

int run_host_until_globally_terminal(HostSccCase& host, std::uint64_t& body_count) {
  body_count = 0u;
  std::string error;
  for (;;) {
    bool any_active = false;
    const auto& state = host.driver_state();
    for (std::int64_t system = 0; system < host.batch_size(); ++system) {
      any_active = any_active || (state.system_statuses[system] == XTBLOOM_STATUS_SUCCESS &&
                                  state.converged[system] == 0u &&
                                  state.iterations[system] < host.options().maximum_iterations);
    }
    if (!any_active) {
      return 0;
    }
    const xtbloom_status_t status = host.run_one_iteration(error);
    if (status != XTBLOOM_STATUS_SUCCESS && status != XTBLOOM_STATUS_SCC_NOT_CONVERGED &&
        status != XTBLOOM_STATUS_EIGENSOLVER_FAILED) {
      std::fprintf(stderr, "CPU conditional reference failed at %llu: status=%d error=%s\n",
                   static_cast<unsigned long long>(body_count), status, error.c_str());
      return __LINE__;
    }
    ++body_count;
  }
}

int compare_graph_loop_cpu_parity(const HostSccCase& host, const Gfn2SccIterationBinding& binding,
                                  cudaStream_t observation_stream) {
  constexpr double kTolerance = 1.0e-8;
  const auto& layout = host.wavefunction_layout();
  const auto& state = binding.state;
  std::vector<double> eigenvalues;
  std::vector<double> occupations;
  std::vector<double> density;
  std::vector<double> weighted_density;
  std::vector<double> qsh;
  std::vector<double> qat;
  std::vector<double> dipoles;
  std::vector<double> quadrupoles;
  std::vector<std::uint64_t> iterations;
  std::vector<xtbloom_status_t> statuses;
  std::vector<std::uint8_t> converged;

  CHECK(download(state.eigenpairs.eigenvalues, state.eigenpairs.eigenvalue_elements, eigenvalues,
                 observation_stream));
  CHECK(download(state.occupations.occupations, state.occupations.occupation_elements, occupations,
                 observation_stream));
  CHECK(
      download(state.density.density, state.density.density_elements, density, observation_stream));
  CHECK(download(state.density.energy_weighted_density, state.density.weighted_density_elements,
                 weighted_density, observation_stream));
  CHECK(download(state.raw_population.qsh, state.raw_population.qsh_elements, qsh,
                 observation_stream));
  CHECK(download(state.raw_population.qat, state.raw_population.qat_elements, qat,
                 observation_stream));
  CHECK(download(state.raw_population.dipole, state.raw_population.dipole_elements, dipoles,
                 observation_stream));
  CHECK(download(state.raw_population.quadrupole, state.raw_population.quadrupole_elements,
                 quadrupoles, observation_stream));
  CHECK(download(state.scc.iterations, state.scc.batch_elements, iterations, observation_stream));
  CHECK(
      download(state.scc.system_statuses, state.scc.batch_elements, statuses, observation_stream));
  CHECK(download(state.scc.converged, state.scc.batch_elements, converged, observation_stream));
  CUDA_CHECK(cudaStreamSynchronize(observation_stream));

  CHECK(compare_doubles("Graph eigenvalues", eigenvalues, host.wavefunction().eigenvalues,
                        layout.eigenvalues.element_count, kTolerance));
  CHECK(compare_doubles("Graph occupations", occupations, host.wavefunction().occupations,
                        layout.occupations.element_count, kTolerance));
  CHECK(compare_doubles("Graph density", density, host.wavefunction().density,
                        layout.density.element_count, kTolerance));
  CHECK(compare_doubles("Graph weighted density", weighted_density,
                        host.wavefunction().energy_weighted_density,
                        layout.energy_weighted_density.element_count, kTolerance));
  CHECK(compare_doubles("Graph qsh", qsh, host.wavefunction().qsh, layout.qsh.element_count,
                        kTolerance));
  CHECK(compare_doubles("Graph qat", qat, host.wavefunction().qat, layout.qat.element_count,
                        kTolerance));
  CHECK(compare_doubles("Graph dipoles", dipoles, host.wavefunction().dipole,
                        layout.dipole.element_count, kTolerance));
  CHECK(compare_doubles("Graph quadrupoles", quadrupoles, host.wavefunction().quadrupole,
                        layout.quadrupole.element_count, kTolerance));
  CHECK(compare_exact_values("Graph iterations", iterations, host.driver_state().iterations,
                             host.batch_size()));
  CHECK(compare_exact_values("Graph statuses", statuses, host.driver_state().system_statuses,
                             host.batch_size()));
  CHECK(compare_exact_values("Graph converged", converged, host.driver_state().converged,
                             host.batch_size()));
  /* One/two-step tests above gate df/u/omega directly. Across eight iterations
   * those private Broyden vectors amplify provider association-order changes;
   * Graph acceptance instead gates every published and convergence observable. */
  CHECK(compare_energy_mixer_and_scc_trace(host, binding.input, binding.state, observation_stream,
                                           kTolerance, 0.0) == 0);
  return 0;
}

int test_conditional_graph_exact_body_count(std::int64_t batch_size,
                                            bool mixed_spin_batch = false) {
  ProductionFixture fixture;
  CHECK(fixture.create(false, batch_size, false, mixed_spin_batch));
  CHECK(fixture.binding.plan.eigensolver_provider.capture_mode ==
        Gfn2SccIterationProviderCaptureMode::kGraphSupported);
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  const HostSccCheckpoint initial = fixture.host.checkpoint();

  Gfn2SccLoopCudaGraphOwner graph;
  const Gfn2SccLoopGraphBuildResult build = graph.build(fixture.binding);
  if (!build.success() || !build.conditional_graph_ready()) {
    std::fprintf(stderr,
                 "conditional SCC graph build failed: status=%u fallback=%u cuda=%d "
                 "iteration=%u stage=%u\n",
                 static_cast<unsigned>(build.status), static_cast<unsigned>(build.fallback_reason),
                 static_cast<int>(build.cuda_status), static_cast<unsigned>(build.iteration.status),
                 static_cast<unsigned>(build.iteration.stage));
  }
  CHECK(build.success());
  CHECK(build.conditional_graph_ready());
  if (!mixed_spin_batch && batch_size == 1) {
    CHECK(build.device_tail_graph_ready());
    CHECK(!build.device_dispatch_chain_ready());
  }
  CHECK(graph.ready());
  CHECK(graph.conditional_graph_ready());
  CHECK(graph.canonical_active_count_device() != nullptr);
  CHECK(graph.numerical_body_count_device() != nullptr);
  CHECK(graph.device_launch_error_device() != nullptr);

  const auto run_and_compare = [&]() -> int {
    const Gfn2SccLoopLaunchResult launch = graph.launch(fixture.handles.stream());
    CHECK(launch.success());
    CHECK(launch.execution_mode == Gfn2SccLoopExecutionMode::kConditionalGraph ||
          launch.execution_mode == Gfn2SccLoopExecutionMode::kDeviceDispatchChain);
    CHECK(launch.submitted_graphs == 1u);
    CHECK(launch.submitted_iterations == 0u);
    std::uint32_t terminal_active_count = 1u;
    std::uint32_t device_launch_error = cudaErrorUnknown;
    std::uint64_t body_count = 0u;
    CHECK(download_value(graph.canonical_active_count_device(), terminal_active_count,
                         fixture.handles.stream()));
    CHECK(
        download_value(graph.numerical_body_count_device(), body_count, fixture.handles.stream()));
    CHECK(download_value(graph.device_launch_error_device(), device_launch_error,
                         fixture.handles.stream()));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

    std::uint64_t reference_body_count = 0u;
    CHECK(run_host_until_globally_terminal(fixture.host, reference_body_count) == 0);
    CHECK(reference_body_count > 0u);
    CHECK(reference_body_count <= fixture.host.options().maximum_iterations);
    CHECK(body_count == reference_body_count);
    CHECK(terminal_active_count == 0u);
    CHECK(device_launch_error == cudaSuccess);
    CHECK(compare_graph_loop_cpu_parity(fixture.host, fixture.binding, fixture.handles.stream()) ==
          0);

    /* A terminal replay must execute no numerical body, proving that the root
     * activity gate suppresses the device-tail body rather than merely gating
     * publication inside an otherwise unconditional iteration. */
    const Gfn2SccLoopLaunchResult terminal = graph.launch(fixture.handles.stream());
    CHECK(terminal.success());
    CHECK(
        download_value(graph.numerical_body_count_device(), body_count, fixture.handles.stream()));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
    CHECK(body_count == 0u);
    return 0;
  };

  CHECK(run_and_compare() == 0);

  std::string error;
  CHECK(fixture.host.restore(initial, error) == XTBLOOM_STATUS_SUCCESS);
  Gfn2SccIterationInitializationReady ready{};
  CHECK(fixture.initializer
            .upload_async(fixture.iteration_arena.get(), fixture.iteration_arena.bytes(), ready,
                          fixture.handles.stream())
            .success());
  CHECK(run_and_compare() == 0);
  return 0;
}

struct DeterministicDebugSnapshot {
  std::vector<double> free_energies;
  std::vector<double> qsh;
  std::vector<double> qat;
  std::vector<double> dipoles;
  std::vector<double> quadrupoles;
  std::vector<std::uint64_t> iterations;
  std::vector<xtbloom_status_t> statuses;
  std::vector<std::uint8_t> converged;
};

bool download_deterministic_debug_snapshot(const Gfn2SccIterationBinding& binding,
                                           cudaStream_t stream,
                                           DeterministicDebugSnapshot& snapshot) {
  const auto& state = binding.state;
  return download(state.scc.free_energies, state.scc.batch_elements, snapshot.free_energies,
                  stream) &&
         download(state.raw_population.qsh, state.raw_population.qsh_elements, snapshot.qsh,
                  stream) &&
         download(state.raw_population.qat, state.raw_population.qat_elements, snapshot.qat,
                  stream) &&
         download(state.raw_population.dipole, state.raw_population.dipole_elements,
                  snapshot.dipoles, stream) &&
         download(state.raw_population.quadrupole, state.raw_population.quadrupole_elements,
                  snapshot.quadrupoles, stream) &&
         download(state.scc.iterations, state.scc.batch_elements, snapshot.iterations, stream) &&
         download(state.scc.system_statuses, state.scc.batch_elements, snapshot.statuses, stream) &&
         download(state.scc.converged, state.scc.batch_elements, snapshot.converged, stream);
}

bool same_deterministic_debug_snapshot(const DeterministicDebugSnapshot& first,
                                       const DeterministicDebugSnapshot& second) {
  return first.free_energies == second.free_energies && first.qsh == second.qsh &&
         first.qat == second.qat && first.dipoles == second.dipoles &&
         first.quadrupoles == second.quadrupoles && first.iterations == second.iterations &&
         first.statuses == second.statuses && first.converged == second.converged;
}

/* Exercise the complete restricted SCC Graph with pedantic cuBLAS math, not
 * merely an isolated eigensolver call. Homogeneous CH2 batches keep this gate
 * bounded while providing a nontrivial 6-AO eigensystem and covering singleton
 * device-tail plus multi-system dispatch-chain execution at every release
 * batch size. */
int test_deterministic_debug_restricted_scc_gate() {
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    ProductionFixture fixture;
    CHECK(fixture.create(false, batch_size, false, false, {SmallSystemKind::kCH2}, 0.0, {}, 64u,
                         1.0e-8, 1.0e-8, true));
    CHECK(fixture.binding.plan.eigensolver_options.deterministic_debug);
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
    const HostSccCheckpoint initial = fixture.host.checkpoint();

    Gfn2SccLoopCudaGraphOwner graph;
    const Gfn2SccLoopGraphBuildResult build = graph.build(fixture.binding);
    CHECK(build.success());
    CHECK(build.conditional_graph_ready());

    const auto run_and_compare = [&](DeterministicDebugSnapshot& snapshot) -> int {
      const Gfn2SccLoopLaunchResult launch = graph.launch(fixture.handles.stream());
      CHECK(launch.success());
      CHECK(download_deterministic_debug_snapshot(fixture.binding, fixture.handles.stream(),
                                                  snapshot));
      CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

      std::uint64_t reference_body_count = 0u;
      CHECK(run_host_until_globally_terminal(fixture.host, reference_body_count) == 0);
      CHECK(reference_body_count > 0u);
      CHECK(std::all_of(snapshot.statuses.begin(), snapshot.statuses.end(),
                        [](xtbloom_status_t status) { return status == XTBLOOM_STATUS_SUCCESS; }));
      CHECK(std::all_of(snapshot.converged.begin(), snapshot.converged.end(),
                        [](std::uint8_t converged) { return converged == 1u; }));
      CHECK(compare_graph_loop_cpu_parity(fixture.host, fixture.binding,
                                          fixture.handles.stream()) == 0);
      return 0;
    };

    DeterministicDebugSnapshot first;
    CHECK(run_and_compare(first) == 0);

    std::string error;
    CHECK(fixture.host.restore(initial, error) == XTBLOOM_STATUS_SUCCESS);
    Gfn2SccIterationInitializationReady ready{};
    CHECK(fixture.initializer
              .upload_async(fixture.iteration_arena.get(), fixture.iteration_arena.bytes(), ready,
                            fixture.handles.stream())
              .success());

    DeterministicDebugSnapshot second;
    CHECK(run_and_compare(second) == 0);
    CHECK(same_deterministic_debug_snapshot(first, second));
  }
  return 0;
}

/* CUDA 12.9 vector-mode XsyevBatched capture succeeds through 512 orbitals and
 * fails at 513. This physical 542-orbital singleton must select the low-level
 * tridiagonal provider so the device-tail Graph stops at convergence instead
 * of submitting all 500 bounded SCC bodies. */
int test_large_singleton_tridiagonal_graph() {
  ProductionFixture fixture;
  CHECK(fixture.create(false, 1, false, false, {SmallSystemKind::kC90H182}, 0.0, {}, 500u, 1.0e-4,
                       1.0e-4));
  CHECK(fixture.host.total_atoms() == 272);
  CHECK(fixture.binding.plan.eigensolver_provider.bucket_count == 1);
  CHECK(fixture.binding.plan.eigensolver_provider.buckets[0].system_count == 1);
  CHECK(fixture.binding.plan.eigensolver_provider.buckets[0].orbital_count == 542);
  CHECK(gfn2_eigensolver_uses_tridiagonal(fixture.binding.plan.eigensolver_options,
                                          fixture.binding.plan.eigensolver_provider.buckets[0]));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  Gfn2SccLoopCudaGraphOwner owner;
  const Gfn2SccLoopGraphBuildResult build = owner.build(fixture.binding);
  CHECK(build.success());
  CHECK(build.device_tail_graph_ready());
  CHECK(build.conditional_graph_ready());
  CHECK(build.fallback_reason == Gfn2SccLoopGraphFallbackReason::kNone);
  CHECK(owner.device_tail_graph_ready());
  CHECK(owner.conditional_graph_ready());
  CHECK(owner.canonical_active_count_device() != nullptr);
  CHECK(owner.numerical_body_count_device() != nullptr);

  const Gfn2SccLoopLaunchResult launch = owner.launch(fixture.handles.stream());
  CHECK(launch.success());
  CHECK(launch.execution_mode == Gfn2SccLoopExecutionMode::kDeviceTailGraph);
  CHECK(launch.submitted_graphs == 1u);
  CHECK(launch.submitted_iterations == 0u);

  std::vector<std::uint64_t> iterations;
  std::vector<std::uint8_t> converged;
  std::vector<xtbloom_status_t> statuses;
  std::vector<double> free_energies;
  std::uint32_t terminal_active_count = 1u;
  std::uint64_t body_count = 0u;
  CHECK(download(fixture.binding.state.scc.iterations, 1, iterations, fixture.handles.stream()));
  CHECK(download(fixture.binding.state.scc.converged, 1, converged, fixture.handles.stream()));
  CHECK(download(fixture.binding.state.scc.system_statuses, 1, statuses, fixture.handles.stream()));
  CHECK(download(fixture.binding.state.scc.free_energies, 1, free_energies,
                 fixture.handles.stream()));
  CHECK(download_value(owner.canonical_active_count_device(), terminal_active_count,
                       fixture.handles.stream()));
  CHECK(download_value(owner.numerical_body_count_device(), body_count, fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  CHECK(iterations[0] > 0u);
  CHECK(iterations[0] < fixture.host.options().maximum_iterations);
  CHECK(converged[0] == 1u);
  CHECK(statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::isfinite(free_energies[0]));
  CHECK(body_count == iterations[0]);
  CHECK(terminal_active_count == 0u);

  const Gfn2SccLoopLaunchResult terminal = owner.launch(fixture.handles.stream());
  CHECK(terminal.success());
  CHECK(terminal.execution_mode == Gfn2SccLoopExecutionMode::kDeviceTailGraph);
  CHECK(terminal.submitted_iterations == 0u);
  CHECK(download_value(owner.numerical_body_count_device(), body_count, fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  CHECK(body_count == 0u);

  /* A deliberately nonconvergent cap proves the device-tail Graph retains the
   * exact public correctness bound instead of relying on the benchmark's
   * normal six-iteration convergence. */
  ProductionFixture limited;
  CHECK(limited.create(false, 1, false, false, {SmallSystemKind::kC90H182}, 0.0, {}, 7u, 1.0e-30,
                       1.0e-30));
  CUDA_CHECK(cudaStreamSynchronize(limited.handles.stream()));
  Gfn2SccLoopCudaGraphOwner limited_owner;
  CHECK(limited_owner.build(limited.binding).device_tail_graph_ready());
  const Gfn2SccLoopLaunchResult limited_launch = limited_owner.launch(limited.handles.stream());
  CHECK(limited_launch.success());
  CHECK(limited_launch.execution_mode == Gfn2SccLoopExecutionMode::kDeviceTailGraph);
  CHECK(limited_launch.submitted_graphs == 1u);
  CHECK(limited_launch.submitted_iterations == 0u);
  CHECK(download(limited.binding.state.scc.iterations, 1, iterations, limited.handles.stream()));
  CHECK(download(limited.binding.state.scc.converged, 1, converged, limited.handles.stream()));
  CHECK(download(limited.binding.state.scc.system_statuses, 1, statuses, limited.handles.stream()));
  CHECK(download_value(limited_owner.numerical_body_count_device(), body_count,
                       limited.handles.stream()));
  CHECK(download_value(limited_owner.canonical_active_count_device(), terminal_active_count,
                       limited.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(limited.handles.stream()));
  CHECK(iterations[0] == 7u);
  CHECK(converged[0] == 0u);
  CHECK(statuses[0] == XTBLOOM_STATUS_SCC_NOT_CONVERGED);
  CHECK(body_count == 7u);
  CHECK(terminal_active_count == 0u);

  /* Unrestricted singletons submit alpha and beta sequentially through the
   * same provider arena. Prove both calls remain device-launchable and retain
   * the composed SCC loop's convergence and publication semantics. */
  ProductionFixture unrestricted;
  CHECK(unrestricted.create(false, 1, true, false, {SmallSystemKind::kC90H182}, 0.0, {}, 50u,
                            1.0e-2, 1.0e-2));
  CHECK(unrestricted.binding.plan.eigensolver_provider.bucket_count == 1);
  CHECK(unrestricted.binding.plan.eigensolver_provider.buckets[0].orbital_count == 542);
  CHECK(unrestricted.binding.plan.eigensolver_provider.buckets[0].system_count == 1);
  CHECK(unrestricted.binding.plan.eigensolver_provider.buckets[0].solve_count == 2);
  CUDA_CHECK(cudaStreamSynchronize(unrestricted.handles.stream()));
  Gfn2SccLoopCudaGraphOwner unrestricted_owner;
  CHECK(unrestricted_owner.build(unrestricted.binding).device_tail_graph_ready());
  const Gfn2SccLoopLaunchResult unrestricted_launch =
      unrestricted_owner.launch(unrestricted.handles.stream());
  CHECK(unrestricted_launch.success());
  CHECK(unrestricted_launch.execution_mode == Gfn2SccLoopExecutionMode::kDeviceTailGraph);
  CHECK(download_value(unrestricted_owner.numerical_body_count_device(), body_count,
                       unrestricted.handles.stream()));
  CHECK(download(unrestricted.binding.state.scc.iterations, 1, iterations,
                 unrestricted.handles.stream()));
  CHECK(download(unrestricted.binding.state.scc.converged, 1, converged,
                 unrestricted.handles.stream()));
  CHECK(download(unrestricted.binding.state.scc.system_statuses, 1, statuses,
                 unrestricted.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(unrestricted.handles.stream()));
  CHECK(body_count == iterations[0]);
  CHECK(iterations[0] > 0u && iterations[0] < 50u);
  CHECK(converged[0] == 1u);
  CHECK(statuses[0] == XTBLOOM_STATUS_SUCCESS);
  return 0;
}

/* Sanitizers instrument every kernel in a device-tail replay and can make the
 * full convergence/terminal/unrestricted matrix prohibitively slow. This
 * one-body production smoke keeps the same 542-orbital provider, setup-owned
 * workspace, and device-launchable SCC Graph while bounding instrumentation. */
int test_large_singleton_tridiagonal_sanitizer_smoke() {
  ProductionFixture fixture;
  CHECK(fixture.create(false, 1, false, false, {SmallSystemKind::kC90H182}, 0.0, {}, 1u, 1.0e-30,
                       1.0e-30));
  CHECK(fixture.binding.plan.eigensolver_provider.bucket_count == 1);
  CHECK(fixture.binding.plan.eigensolver_provider.buckets[0].orbital_count == 542);
  CHECK(gfn2_eigensolver_uses_tridiagonal(fixture.binding.plan.eigensolver_options,
                                          fixture.binding.plan.eigensolver_provider.buckets[0]));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  Gfn2SccLoopCudaGraphOwner owner;
  CHECK(owner.build(fixture.binding).device_tail_graph_ready());
  const Gfn2SccLoopLaunchResult launch = owner.launch(fixture.handles.stream());
  CHECK(launch.success());
  CHECK(launch.execution_mode == Gfn2SccLoopExecutionMode::kDeviceTailGraph);
  std::uint64_t body_count = 0u;
  std::vector<std::uint64_t> iterations;
  std::vector<xtbloom_status_t> statuses;
  CHECK(download_value(owner.numerical_body_count_device(), body_count, fixture.handles.stream()));
  CHECK(download(fixture.binding.state.scc.iterations, 1, iterations, fixture.handles.stream()));
  CHECK(download(fixture.binding.state.scc.system_statuses, 1, statuses, fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  CHECK(body_count == 1u);
  CHECK(iterations == std::vector<std::uint64_t>{1u});
  CHECK(statuses == std::vector<xtbloom_status_t>{XTBLOOM_STATUS_SCC_NOT_CONVERGED});
  return 0;
}

/* Forced exact-capacity dispatch-chain build, launch, terminal replay, and
 * CPU parity for restricted batches. Preferring kDeviceDispatchChain must
 * produce a ready chain whose executable count equals the documented table
 * layout: pre + post + sum over buckets of [cap eig + (cap+1) back].
 * Launch must report the dispatch-chain mode and every observable must match
 * the sequential CPU reference, including a WARM restart after restoring the
 * initial checkpoint. */
int test_dispatch_chain_forced_build_and_parity(std::int64_t batch_size) {
  ProductionFixture fixture;
  CHECK(fixture.create(false, batch_size, false, /*mixed_spin_batch=*/false));
  CHECK(fixture.binding.plan.eigensolver_provider.capture_mode ==
        Gfn2SccIterationProviderCaptureMode::kGraphSupported);
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  const HostSccCheckpoint initial = fixture.host.checkpoint();

  const std::int64_t bucket_count = fixture.binding.plan.eigensolver_provider.bucket_count;
  CHECK(bucket_count > 0);
  std::int64_t expected_executables = 2;
  for (std::int64_t b = 0; b < bucket_count; ++b) {
    const std::int64_t cap = fixture.binding.plan.eigensolver_provider.buckets[b].system_count;
    CHECK(cap > 0);
    expected_executables += 2 * cap + 1;
  }

  Gfn2SccLoopCudaGraphOwner graph;
  const Gfn2SccLoopGraphBuildResult build =
      graph.build(fixture.binding, Gfn2SccLoopGraphPreference::kDeviceDispatchChain);
  if (!build.device_dispatch_chain_ready()) {
    std::fprintf(stderr,
                 "forced dispatch-chain build failed: status=%u fallback=%u cuda=%d "
                 "iteration=%u stage=%u\n",
                 static_cast<unsigned>(build.status), static_cast<unsigned>(build.fallback_reason),
                 static_cast<int>(build.cuda_status), static_cast<unsigned>(build.iteration.status),
                 static_cast<unsigned>(build.iteration.stage));
  }
  CHECK(build.device_dispatch_chain_ready());
  CHECK(build.conditional_graph_ready());
  CHECK(graph.ready());
  CHECK(graph.device_dispatch_chain_ready());
  CHECK(graph.conditional_graph_ready());
  CHECK(graph.dispatch_chain_executable_count() == static_cast<std::size_t>(expected_executables));
  CHECK(graph.retained_device_bytes() > 0u);

  const auto run_and_compare = [&]() -> int {
    const Gfn2SccLoopLaunchResult launch = graph.launch(fixture.handles.stream());
    CHECK(launch.success());
    CHECK(launch.execution_mode == Gfn2SccLoopExecutionMode::kDeviceDispatchChain);
    CHECK(launch.submitted_graphs == 1u);
    CHECK(launch.submitted_iterations == 0u);
    std::uint32_t terminal_active_count = 1u;
    std::uint32_t device_launch_error = cudaErrorUnknown;
    std::uint64_t body_count = 0u;
    CHECK(download_value(graph.canonical_active_count_device(), terminal_active_count,
                         fixture.handles.stream()));
    CHECK(
        download_value(graph.numerical_body_count_device(), body_count, fixture.handles.stream()));
    CHECK(download_value(graph.device_launch_error_device(), device_launch_error,
                         fixture.handles.stream()));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

    std::uint64_t reference_body_count = 0u;
    CHECK(run_host_until_globally_terminal(fixture.host, reference_body_count) == 0);
    CHECK(reference_body_count > 0u);
    CHECK(reference_body_count <= fixture.host.options().maximum_iterations);
    CHECK(body_count == reference_body_count);
    CHECK(terminal_active_count == 0u);
    CHECK(device_launch_error == cudaSuccess);
    CHECK(compare_graph_loop_cpu_parity(fixture.host, fixture.binding, fixture.handles.stream()) ==
          0);

    /* A terminal replay must execute no numerical body: the root activity gate
     * suppresses the dispatch-chain pre-executable just as it does the
     * monolithic device-tail body. */
    const Gfn2SccLoopLaunchResult terminal = graph.launch(fixture.handles.stream());
    CHECK(terminal.success());
    CHECK(
        download_value(graph.numerical_body_count_device(), body_count, fixture.handles.stream()));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
    CHECK(body_count == 0u);
    return 0;
  };

  CHECK(run_and_compare() == 0);

  std::string error;
  CHECK(fixture.host.restore(initial, error) == XTBLOOM_STATUS_SUCCESS);
  Gfn2SccIterationInitializationReady ready{};
  CHECK(fixture.initializer
            .upload_async(fixture.iteration_arena.get(), fixture.iteration_arena.bytes(), ready,
                          fixture.handles.stream())
            .success());
  CHECK(run_and_compare() == 0);
  return 0;
}

/* Preference must be observable, not just correct: forced kDeviceTailGraph
 * must never silently upgrade to the dispatch chain, and a forced
 * kDeviceDispatchChain must never silently degrade to the monolithic
 * device-tail graph. */
int test_dispatch_chain_preference_is_honored() {
  ProductionFixture fixture;
  CHECK(fixture.create(false, 8, false, /*mixed_spin_batch=*/false));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  {
    Gfn2SccLoopCudaGraphOwner graph;
    const Gfn2SccLoopGraphBuildResult build =
        graph.build(fixture.binding, Gfn2SccLoopGraphPreference::kDeviceTailGraph);
    CHECK(build.device_tail_graph_ready());
    CHECK(!build.device_dispatch_chain_ready());
    CHECK(graph.device_tail_graph_ready());
    CHECK(!graph.device_dispatch_chain_ready());
    CHECK(graph.dispatch_chain_executable_count() == 0u);
    const Gfn2SccLoopLaunchResult launch = graph.launch(fixture.handles.stream());
    CHECK(launch.success());
    CHECK(launch.execution_mode == Gfn2SccLoopExecutionMode::kDeviceTailGraph);
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  }

  {
    Gfn2SccLoopCudaGraphOwner graph;
    const Gfn2SccLoopGraphBuildResult build =
        graph.build(fixture.binding, Gfn2SccLoopGraphPreference::kDeviceDispatchChain);
    CHECK(build.device_dispatch_chain_ready());
    CHECK(!build.device_tail_graph_ready());
    CHECK(graph.device_dispatch_chain_ready());
    CHECK(!graph.device_tail_graph_ready());
    CHECK(graph.dispatch_chain_executable_count() > 0u);
    const Gfn2SccLoopLaunchResult launch = graph.launch(fixture.handles.stream());
    CHECK(launch.success());
    CHECK(launch.execution_mode == Gfn2SccLoopExecutionMode::kDeviceDispatchChain);
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  }
  return 0;
}

/* Mixed-spin batches cannot use the restricted compaction map, so a forced
 * dispatch-chain build must return the bounded fallback with an observable
 * dispatch-specific reason rather than silently producing a device-tail graph
 * or a chain it cannot support. Auto must still prefer the monolithic
 * device-tail graph for mixed-spin batches. */
int test_dispatch_chain_mixed_spin_falls_back() {
  ProductionFixture fixture;
  CHECK(fixture.create(false, 8, false, /*mixed_spin_batch=*/true));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  {
    Gfn2SccLoopCudaGraphOwner graph;
    const Gfn2SccLoopGraphBuildResult build =
        graph.build(fixture.binding, Gfn2SccLoopGraphPreference::kDeviceDispatchChain);
    CHECK(build.device_dispatch_chain_ready() == false);
    CHECK(build.device_tail_graph_ready() == false);
    CHECK(build.success());
    CHECK(build.status == Gfn2SccLoopGraphBuildStatus::kBoundedFallbackReady);
    CHECK(build.fallback_reason == Gfn2SccLoopGraphFallbackReason::kDispatchUnsupportedLayout);
    CHECK(!graph.device_dispatch_chain_ready());
    CHECK(graph.dispatch_chain_executable_count() == 0u);
  }

  {
    Gfn2SccLoopCudaGraphOwner graph;
    const Gfn2SccLoopGraphBuildResult build =
        graph.build(fixture.binding, Gfn2SccLoopGraphPreference::kAuto);
    CHECK(build.device_tail_graph_ready());
    CHECK(!build.device_dispatch_chain_ready());
    CHECK(graph.device_tail_graph_ready());
    CHECK(!graph.device_dispatch_chain_ready());
  }
  return 0;
}

/* The production kAuto preference must choose the monolithic device-tail
 * graph whenever the batch's largest eigensolver bucket exceeds the measured
 * dispatch-chain regime, and must keep preferring the exact-capacity chain for
 * the small-AO batches that issue #80/#131 validated. Forced preferences are
 * unaffected by the regime predicate. */
int test_kauto_dispatch_chain_regime_selection() {
  ProductionFixture fixture;
  CHECK(fixture.create(false, 8, false, /*mixed_spin_batch=*/false));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  CHECK(fixture.binding.plan.eigensolver_provider.capture_mode ==
        Gfn2SccIterationProviderCaptureMode::kGraphSupported);

  Gfn2SccIterationDevicePlan plan = fixture.binding.plan;
  Gfn2EigensolverBucket synthetic_buckets[4]{};
  plan.eigensolver_provider.buckets = synthetic_buckets;

  plan.eigensolver_provider.bucket_count = 0;
  CHECK(!gfn2_scc_dispatch_chain_regime_applies(plan));

  synthetic_buckets[0].orbital_count = 122;
  plan.eigensolver_provider.bucket_count = 1;
  CHECK(!gfn2_scc_dispatch_chain_regime_applies(plan));

  synthetic_buckets[0].orbital_count = 40;
  CHECK(gfn2_scc_dispatch_chain_regime_applies(plan));

  synthetic_buckets[0].orbital_count = 41;
  CHECK(!gfn2_scc_dispatch_chain_regime_applies(plan));

  synthetic_buckets[0].orbital_count = 40;
  synthetic_buckets[1].orbital_count = 20;
  plan.eigensolver_provider.bucket_count = 2;
  CHECK(gfn2_scc_dispatch_chain_regime_applies(plan));

  synthetic_buckets[1].orbital_count = 122;
  CHECK(!gfn2_scc_dispatch_chain_regime_applies(plan));

  plan.eigensolver_provider.bucket_count = 1;
  plan.topology.batch_size = 1;
  synthetic_buckets[0].system_count = 1;
  synthetic_buckets[0].orbital_count = 512;
  CHECK(!gfn2_eigensolver_uses_tridiagonal(plan.eigensolver_options, synthetic_buckets[0]));
  synthetic_buckets[0].orbital_count = 513;
  CHECK(gfn2_eigensolver_uses_tridiagonal(plan.eigensolver_options, synthetic_buckets[0]));
  synthetic_buckets[0].system_count = 2;
  CHECK(!gfn2_eigensolver_uses_tridiagonal(plan.eigensolver_options, synthetic_buckets[0]));
  synthetic_buckets[0].system_count = 1;
  plan.topology.batch_size = 2;
  CHECK(gfn2_eigensolver_uses_tridiagonal(plan.eigensolver_options, synthetic_buckets[0]));
  plan.topology.batch_size = 1;
  synthetic_buckets[0].orbital_count = kGfn2TridiagonalMaximumOrbitals + 1;
  CHECK(!gfn2_eigensolver_uses_tridiagonal(plan.eigensolver_options, synthetic_buckets[0]));
  plan = fixture.binding.plan;

  /* The real small-AO fixture must still prefer the chain under kAuto. */
  CHECK(gfn2_scc_dispatch_chain_regime_applies(fixture.binding.plan));
  Gfn2SccLoopCudaGraphOwner auto_graph;
  const Gfn2SccLoopGraphBuildResult auto_build =
      auto_graph.build(fixture.binding, Gfn2SccLoopGraphPreference::kAuto);
  CHECK(auto_build.device_dispatch_chain_ready());
  CHECK(!auto_build.device_tail_graph_ready());
  return 0;
}

/* The production runtime builds the SCC loop through the geometry-epoch
 * overload (Gfn2GeometryEpochConsumerDevice), which selects the epoch
 * prepare/compact variant and the dynamic-geometry pre/post segment launchers.
 * Exercise that exact path against the forced dispatch chain: the chain must
 * build ready with the expected executable count, run to global terminal with
 * CPU parity, and survive a WARM restart through the same epoch-owning path. */
int test_dispatch_chain_dynamic_geometry_epoch_parity() {
  constexpr std::int64_t batch_size = 8;
  ProductionFixture fixture;
  CHECK(fixture.create(false, batch_size, false, /*mixed_spin_batch=*/false));
  CHECK(fixture.binding.plan.eigensolver_provider.capture_mode ==
        Gfn2SccIterationProviderCaptureMode::kGraphSupported);
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  const HostSccCheckpoint initial = fixture.host.checkpoint();

  DeviceAllocation epoch_storage;
  DeviceAllocation eligible_storage;
  CHECK(epoch_storage.allocate(sizeof(std::uint64_t)));
  CHECK(eligible_storage.allocate(static_cast<std::size_t>(batch_size) * sizeof(std::uint8_t)));
  auto* const epoch = static_cast<std::uint64_t*>(epoch_storage.get());
  auto* const eligible = static_cast<std::uint8_t*>(eligible_storage.get());
  const std::uint64_t* const committed = fixture.binding.plan.geometry_cache.geometry_generations;
  std::vector<std::uint64_t> generations(static_cast<std::size_t>(batch_size), kGeometryGeneration);
  std::vector<std::uint8_t> eligibility(static_cast<std::size_t>(batch_size), 1u);
  std::vector<Gfn2SccCacheProvenanceBinding> provenance;
  CHECK(download(fixture.binding.plan.provenance.cache_bindings,
                 fixture.binding.plan.provenance.cache_binding_count, provenance,
                 fixture.handles.stream()));
  for (auto& record : provenance) {
    record.provenance.generation_scope = Gfn2GenerationScope::kPerSystem;
    record.provenance.geometry_generation = 0u;
    record.provenance.system_generation_count = batch_size;
    record.provenance.system_geometry_generations = committed;
  }
  CUDA_CHECK(cudaMemcpyAsync(
      const_cast<Gfn2SccCacheProvenanceBinding*>(fixture.binding.plan.provenance.cache_bindings),
      provenance.data(), provenance.size() * sizeof(Gfn2SccCacheProvenanceBinding),
      cudaMemcpyHostToDevice, fixture.handles.stream()));
  CUDA_CHECK(cudaMemcpyAsync(epoch, &generations[0], sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                             fixture.handles.stream()));
  CUDA_CHECK(cudaMemcpyAsync(const_cast<std::uint64_t*>(committed), generations.data(),
                             generations.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                             fixture.handles.stream()));
  CUDA_CHECK(cudaMemcpyAsync(eligible, eligibility.data(), eligibility.size(),
                             cudaMemcpyHostToDevice, fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  const Gfn2GeometryEpochConsumerDevice consumer{
      {epoch, 1, kPlanToken}, committed, eligible, batch_size, kPlanToken};

  const std::int64_t bucket_count = fixture.binding.plan.eigensolver_provider.bucket_count;
  std::int64_t expected_executables = 2;
  for (std::int64_t b = 0; b < bucket_count; ++b) {
    expected_executables +=
        2 * fixture.binding.plan.eigensolver_provider.buckets[b].system_count + 1;
  }

  Gfn2SccLoopCudaGraphOwner graph;
  const Gfn2SccLoopGraphBuildResult build =
      graph.build(fixture.binding, consumer, Gfn2SccLoopGraphPreference::kDeviceDispatchChain);
  if (!build.device_dispatch_chain_ready()) {
    std::fprintf(stderr,
                 "epoch dispatch-chain build failed: status=%u fallback=%u cuda=%d "
                 "iteration=%u stage=%u\n",
                 static_cast<unsigned>(build.status), static_cast<unsigned>(build.fallback_reason),
                 static_cast<int>(build.cuda_status), static_cast<unsigned>(build.iteration.status),
                 static_cast<unsigned>(build.iteration.stage));
  }
  CHECK(build.device_dispatch_chain_ready());
  CHECK(graph.device_dispatch_chain_ready());
  CHECK(graph.dispatch_chain_executable_count() == static_cast<std::size_t>(expected_executables));

  const auto run_and_compare = [&]() -> int {
    const Gfn2SccLoopLaunchResult launch = graph.launch(fixture.handles.stream());
    CHECK(launch.success());
    CHECK(launch.execution_mode == Gfn2SccLoopExecutionMode::kDeviceDispatchChain);
    std::uint32_t device_launch_error = cudaErrorUnknown;
    std::uint64_t body_count = 0u;
    CHECK(download_value(graph.device_launch_error_device(), device_launch_error,
                         fixture.handles.stream()));
    CHECK(
        download_value(graph.numerical_body_count_device(), body_count, fixture.handles.stream()));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
    std::uint64_t reference_body_count = 0u;
    CHECK(run_host_until_globally_terminal(fixture.host, reference_body_count) == 0);
    CHECK(reference_body_count > 0u);
    CHECK(reference_body_count <= fixture.host.options().maximum_iterations);
    CHECK(body_count == reference_body_count);
    CHECK(device_launch_error == cudaSuccess);
    CHECK(compare_graph_loop_cpu_parity(fixture.host, fixture.binding, fixture.handles.stream()) ==
          0);
    return 0;
  };

  CHECK(run_and_compare() == 0);

  std::string error;
  CHECK(fixture.host.restore(initial, error) == XTBLOOM_STATUS_SUCCESS);
  Gfn2SccIterationInitializationReady ready{};
  CHECK(fixture.initializer
            .upload_async(fixture.iteration_arena.get(), fixture.iteration_arena.bytes(), ready,
                          fixture.handles.stream())
            .success());
  CHECK(run_and_compare() == 0);
  return 0;
}

/* The dispatch-chain owner must behave like the device-tail owner when a
 * caller captures the stream: launch() falls back to the bounded DAG so the
 * outer executable remains valid after owner reset. Exercise that path with a
 * forced dispatch-chain build to prove the chain does not regress whole-
 * pipeline capture. */
int test_dispatch_chain_owner_whole_pipeline_capture() {
  constexpr double kTerminalTolerance = 1.0e-8;
  ProductionFixture fixture;
  CHECK(fixture.create(false, 8, false, /*mixed_spin_batch=*/false));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  GraphResources pipeline;
  DeviceAllocation terminal_snapshot_storage;
  CHECK(terminal_snapshot_storage.allocate(static_cast<std::size_t>(fixture.host.batch_size()) *
                                           sizeof(double)));
  auto* const terminal_snapshot = static_cast<double*>(terminal_snapshot_storage.get());
  {
    Gfn2SccLoopCudaGraphOwner owner;
    const Gfn2SccLoopGraphBuildResult build =
        owner.build(fixture.binding, Gfn2SccLoopGraphPreference::kDeviceDispatchChain);
    CHECK(build.device_dispatch_chain_ready());
    CHECK(owner.device_dispatch_chain_ready());

    /* The chain's device-table executables reference chain-owned control
     * storage. Capture the bounded DAG so the outer executable remains valid
     * after owner reset. */
    CUDA_CHECK(cudaStreamBeginCapture(fixture.handles.stream(), cudaStreamCaptureModeThreadLocal));
    const Gfn2SccLoopLaunchResult captured = owner.launch(fixture.handles.stream());
    CHECK(captured.success());
    CHECK(captured.execution_mode == Gfn2SccLoopExecutionMode::kBoundedFallback);
    CHECK(captured.submitted_graphs == 0u);
    CHECK(captured.submitted_iterations == fixture.host.options().maximum_iterations);
    CUDA_CHECK(cudaMemcpyAsync(terminal_snapshot, fixture.binding.state.scc.free_energies,
                               static_cast<std::size_t>(fixture.host.batch_size()) * sizeof(double),
                               cudaMemcpyDeviceToDevice, fixture.handles.stream()));
    CUDA_CHECK(cudaStreamEndCapture(fixture.handles.stream(), pipeline.graph_address()));
    owner.reset();
    CHECK(!owner.ready());
  }
  CHECK(pipeline.graph() != nullptr);
  CUDA_CHECK(cudaGraphInstantiate(pipeline.executable_address(), pipeline.graph(), 0u));

  CUDA_CHECK(cudaGraphLaunch(pipeline.executable(), fixture.handles.stream()));
  std::vector<double> captured_terminal_free_energies;
  CHECK(download(terminal_snapshot, fixture.host.batch_size(), captured_terminal_free_energies,
                 fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  CHECK(run_host_fixed_scc_loop(fixture.host) == 0);
  CHECK(compare_doubles("captured terminal free energy", captured_terminal_free_energies,
                        fixture.host.driver_state().free_energies, fixture.host.batch_size(),
                        kTerminalTolerance));
  CHECK(compare_graph_loop_cpu_parity(fixture.host, fixture.binding, fixture.handles.stream()) ==
        0);
  return 0;
}

int test_device_tail_owner_whole_pipeline_capture() {
  constexpr double kTerminalTolerance = 1.0e-8;
  ProductionFixture fixture;
  /* Dispatch-chain builds reject mixed-spin batches, so a mixed-spin fixture
   * deterministically exercises the monolithic device-tail fallback that this
   * test (and the bounded whole-pipeline capture) must cover. */
  CHECK(fixture.create(false, 8, false, /*mixed_spin_batch=*/true));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  GraphResources pipeline;
  DeviceAllocation terminal_snapshot_storage;
  CHECK(terminal_snapshot_storage.allocate(static_cast<std::size_t>(fixture.host.batch_size()) *
                                           sizeof(double)));
  auto* const terminal_snapshot = static_cast<double*>(terminal_snapshot_storage.get());
  {
    Gfn2SccLoopCudaGraphOwner owner;
    const Gfn2SccLoopGraphBuildResult build = owner.build(fixture.binding);
    CHECK(build.device_tail_graph_ready());
    CHECK(owner.device_tail_graph_ready());

    /* The device-tail Graph references owner control storage. Capture the
     * bounded DAG so the outer executable remains valid after owner reset. */
    CUDA_CHECK(cudaStreamBeginCapture(fixture.handles.stream(), cudaStreamCaptureModeThreadLocal));
    const Gfn2SccLoopLaunchResult captured = owner.launch(fixture.handles.stream());
    CHECK(captured.success());
    CHECK(captured.execution_mode == Gfn2SccLoopExecutionMode::kBoundedFallback);
    CHECK(captured.submitted_graphs == 0u);
    CHECK(captured.submitted_iterations == fixture.host.options().maximum_iterations);
    CUDA_CHECK(cudaMemcpyAsync(terminal_snapshot, fixture.binding.state.scc.free_energies,
                               static_cast<std::size_t>(fixture.host.batch_size()) * sizeof(double),
                               cudaMemcpyDeviceToDevice, fixture.handles.stream()));
    CUDA_CHECK(cudaStreamEndCapture(fixture.handles.stream(), pipeline.graph_address()));
    owner.reset();
    CHECK(!owner.ready());
  }
  CHECK(pipeline.graph() != nullptr);
  CUDA_CHECK(cudaGraphInstantiate(pipeline.executable_address(), pipeline.graph(), 0u));

  CUDA_CHECK(cudaGraphLaunch(pipeline.executable(), fixture.handles.stream()));
  std::vector<double> captured_terminal_free_energies;
  CHECK(download(terminal_snapshot, fixture.host.batch_size(), captured_terminal_free_energies,
                 fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  CHECK(run_host_fixed_scc_loop(fixture.host) == 0);
  CHECK(compare_doubles("captured terminal free energy", captured_terminal_free_energies,
                        fixture.host.driver_state().free_energies, fixture.host.batch_size(),
                        kTerminalTolerance));
  CHECK(compare_graph_loop_cpu_parity(fixture.host, fixture.binding, fixture.handles.stream()) ==
        0);
  return 0;
}

int test_conditional_graph_plan_failure_forces_exit() {
  ProductionFixture fixture;
  CHECK(fixture.create(false, 8));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  /* Keep root provenance valid but make the ES2 numerical descriptor stale.
   * The first body therefore records a plan-wide ES2 failure before state
   * publication. A device-tail loop that blindly re-derived activity would
   * reset that record and self-launch until the bound; the failure snapshot
   * must stop at exactly one body and restore the original diagnostic. */
  Gfn2SccIterationDevicePlan stale_plan = fixture.binding.plan;
  stale_plan.es2_cache.geometry_generation = kGeometryGeneration - 1u;
  Gfn2SccIterationBinding stale_binding{};
  CHECK(bind_gfn2_scc_iteration_cuda(stale_plan, fixture.binding.input, fixture.binding.state,
                                     fixture.binding.workspace, stale_binding)
            .error == Gfn2SccIterationBindingError::kSuccess);

  Gfn2SccLoopCudaGraphOwner graph;
  const Gfn2SccLoopGraphBuildResult build = graph.build(stale_binding);
  CHECK(build.conditional_graph_ready());
  CHECK(graph.launch(fixture.handles.stream()).success());

  std::uint64_t body_count = 0u;
  std::uint64_t plan_failure = 0u;
  std::uint32_t sequence_active = 1u;
  std::vector<std::uint64_t> iterations;
  CHECK(download_value(graph.numerical_body_count_device(), body_count, fixture.handles.stream()));
  CHECK(download_value(stale_binding.workspace.ledger.plan_failure_record, plan_failure,
                       fixture.handles.stream()));
  CHECK(download_value(stale_binding.workspace.ledger.sequence_active, sequence_active,
                       fixture.handles.stream()));
  CHECK(download(stale_binding.state.scc.iterations, stale_binding.state.scc.batch_elements,
                 iterations, fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  CHECK(body_count == 1u);
  CHECK(sequence_active == 0u);
  CHECK(plan_failure == gfn2_scc_stage_failure_record(
                            Gfn2SccStageId::kES2Potential,
                            static_cast<std::uint32_t>(Gfn2ES2DeviceError::kInvalidCacheMatrix)));
  CHECK(std::all_of(iterations.begin(), iterations.end(),
                    [](std::uint64_t value) { return value == 0u; }));
  return 0;
}

int test_conditional_owner_bounded_fallback_parity(bool mixed_spin_batch = false) {
  ProductionFixture fixture;
  CHECK(fixture.create(false, 8, false, mixed_spin_batch));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  Gfn2SccIterationDevicePlan fallback_plan = fixture.binding.plan;
  fallback_plan.eigensolver_provider.capture_mode =
      Gfn2SccIterationProviderCaptureMode::kUncapturedSegmentRequired;
  Gfn2SccIterationBinding fallback_binding{};
  CHECK(bind_gfn2_scc_iteration_cuda(fallback_plan, fixture.binding.input, fixture.binding.state,
                                     fixture.binding.workspace, fallback_binding)
            .error == Gfn2SccIterationBindingError::kSuccess);

  Gfn2SccLoopCudaGraphOwner owner;
  const Gfn2SccLoopGraphBuildResult build = owner.build(fallback_binding);
  CHECK(build.success());
  CHECK(!build.conditional_graph_ready());
  CHECK(build.fallback_reason == Gfn2SccLoopGraphFallbackReason::kProviderCaptureUnsupported);
  const Gfn2SccLoopLaunchResult launch = owner.launch(fixture.handles.stream());
  CHECK(launch.success());
  CHECK(launch.execution_mode == Gfn2SccLoopExecutionMode::kBoundedFallback);
  CHECK(launch.submitted_iterations == fixture.host.options().maximum_iterations);
  CHECK(run_host_fixed_scc_loop(fixture.host) == 0);
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  CHECK(compare_graph_loop_cpu_parity(fixture.host, fallback_binding, fixture.handles.stream()) ==
        0);
  return 0;
}

int test_conditional_graph_mixed_warm_peer_parity() {
  constexpr std::int64_t kBatch = 8;
  ProductionFixture fixture;
  CHECK(fixture.create(false, kBatch, false, true));
  Gfn2SccLoopCudaGraphOwner graph;
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  CHECK(graph.build(fixture.binding).conditional_graph_ready());

  std::string error;
  for (int iteration = 0; iteration < 2; ++iteration) {
    CHECK(launch_gfn2_restricted_scc_iteration_cuda(fixture.binding, fixture.handles.stream())
              .success());
    CHECK(fixture.host.run_one_iteration(error) == XTBLOOM_STATUS_SUCCESS);
  }
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  auto& driver = fixture.host.driver_state();
  auto& mixer = fixture.host.mixer_state();
  driver.converged[1] = 1u;
  mixer.converged[1] = 1u;
  /* A forced warm/converged peer must retain a self-consistent convergence
   * record. The device-tail body skips this peer, so seed both public SCC and
   * mixer residual views with values that satisfy the bound tolerances. */
  mixer.residual_rms[1] = 0.0;
  mixer.residual_maximum[1] = 0.0;
  driver.system_statuses[2] = XTBLOOM_STATUS_INTERNAL_ERROR;
  mixer.system_statuses[2] = XTBLOOM_STATUS_INTERNAL_ERROR;
  driver.iterations[3] = fixture.host.options().maximum_iterations;
  driver.system_statuses[3] = XTBLOOM_STATUS_SCC_NOT_CONVERGED;
  mixer.iterations[3] = fixture.host.options().maximum_iterations;
  mixer.system_statuses[3] = XTBLOOM_STATUS_SCC_NOT_CONVERGED;

  CUDA_CHECK(cudaMemcpyAsync(fixture.binding.state.scc.iterations, driver.iterations,
                             kBatch * sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                             fixture.handles.stream()));
  CUDA_CHECK(cudaMemcpyAsync(fixture.binding.state.scc.system_statuses, driver.system_statuses,
                             kBatch * sizeof(xtbloom_status_t), cudaMemcpyHostToDevice,
                             fixture.handles.stream()));
  CUDA_CHECK(cudaMemcpyAsync(fixture.binding.state.scc.converged, driver.converged,
                             kBatch * sizeof(std::uint8_t), cudaMemcpyHostToDevice,
                             fixture.handles.stream()));
  CUDA_CHECK(cudaMemcpyAsync(fixture.binding.state.mixer.iterations, mixer.iterations,
                             kBatch * sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                             fixture.handles.stream()));
  CUDA_CHECK(cudaMemcpyAsync(fixture.binding.state.mixer.system_statuses, mixer.system_statuses,
                             kBatch * sizeof(xtbloom_status_t), cudaMemcpyHostToDevice,
                             fixture.handles.stream()));
  CUDA_CHECK(cudaMemcpyAsync(fixture.binding.state.mixer.residual_converged, mixer.converged,
                             kBatch * sizeof(std::uint8_t), cudaMemcpyHostToDevice,
                             fixture.handles.stream()));
  CUDA_CHECK(cudaMemcpyAsync(fixture.binding.state.mixer.residual_rms, mixer.residual_rms,
                             kBatch * sizeof(double), cudaMemcpyHostToDevice,
                             fixture.handles.stream()));
  CUDA_CHECK(cudaMemcpyAsync(fixture.binding.state.mixer.residual_maximum, mixer.residual_maximum,
                             kBatch * sizeof(double), cudaMemcpyHostToDevice,
                             fixture.handles.stream()));
  CUDA_CHECK(cudaMemcpyAsync(fixture.binding.state.scc.residual_rms, mixer.residual_rms,
                             kBatch * sizeof(double), cudaMemcpyHostToDevice,
                             fixture.handles.stream()));

  CHECK(graph.launch(fixture.handles.stream()).success());
  std::uint64_t body_count = 0u;
  CHECK(download_value(graph.numerical_body_count_device(), body_count, fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  std::uint64_t reference_body_count = 0u;
  CHECK(run_host_until_globally_terminal(fixture.host, reference_body_count) == 0);
  CHECK(body_count == reference_body_count);
  CHECK(compare_graph_loop_cpu_parity(fixture.host, fixture.binding, fixture.handles.stream()) ==
        0);
  CHECK(driver.converged[1] == 1u);
  CHECK(driver.system_statuses[2] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(driver.iterations[3] == fixture.host.options().maximum_iterations);
  CHECK(driver.system_statuses[3] == XTBLOOM_STATUS_SCC_NOT_CONVERGED);
  return 0;
}

/* End-to-end production-loop comparison of the exact-capacity device dispatch
 * chain (kDeviceDispatchChain) against the monolithic full-capacity device-tail
 * graph (kDeviceTailGraph) that it replaced as the production default. Both
 * families launch from the identical binding and state, run the identical full
 * SCC loop to global terminal, and must publish identical results. A
 * deterministic per-system terminal ladder (iterations seeded toward the
 * configured maximum, exactly like the reusable real-GPU no-resume ladder) is
 * used to create the mixed-activity buffers where the exact-capacity chain is
 * expected to win by keeping terminal peers out of provider arithmetic. */
int benchmark_dispatch_chain_vs_monolithic() {
  constexpr int kWarmup = 3;
  constexpr int kSamples = 50;
  struct BenchmarkState {
    std::vector<double> free_energies;
    std::vector<double> eigenvalues;
    std::vector<double> occupations;
    std::vector<double> density;
    std::vector<double> weighted_density;
    std::vector<double> qsh;
    std::vector<double> qat;
    std::vector<double> dipoles;
    std::vector<double> quadrupoles;
    std::vector<std::uint64_t> iterations;
    std::vector<xtbloom_status_t> statuses;
    std::vector<std::uint8_t> converged;
  };
  std::printf(
      "{\"record_type\":\"protocol\",\"benchmark\":\"production-dispatch-chain-vs-monolithic\","
      "\"warmups\":%d,\"samples\":%d,\"timing\":\"cudaEvent elapsed full loop to global "
      "terminal on the fixture stream\",\"workloads\":\"heterogeneous small restricted "
      "systems with a per-system terminal ladder (active_fraction 1, 1/2, 1/4)\"}\n",
      kWarmup, kSamples);
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    ProductionFixture fixture;
    CHECK(fixture.create(false, batch_size));
    CHECK(fixture.binding.plan.eigensolver_provider.capture_mode ==
          Gfn2SccIterationProviderCaptureMode::kGraphSupported);
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
    const std::uint64_t maximum = fixture.host.options().maximum_iterations;

    Gfn2SccLoopCudaGraphOwner chain;
    CHECK(chain.build(fixture.binding, Gfn2SccLoopGraphPreference::kDeviceDispatchChain)
              .device_dispatch_chain_ready());
    Gfn2SccLoopCudaGraphOwner monolithic;
    CHECK(monolithic.build(fixture.binding, Gfn2SccLoopGraphPreference::kDeviceTailGraph)
              .device_tail_graph_ready());
    const HostSccCheckpoint initial = fixture.host.checkpoint();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const auto download_state = [&](BenchmarkState& snapshot) -> int {
      const Gfn2SccIterationDeviceState& state = fixture.binding.state;
      CHECK(download(state.scc.free_energies, state.scc.batch_elements, snapshot.free_energies,
                     fixture.handles.stream()));
      CHECK(download(state.eigenpairs.eigenvalues, state.eigenpairs.eigenvalue_elements,
                     snapshot.eigenvalues, fixture.handles.stream()));
      CHECK(download(state.occupations.occupations, state.occupations.occupation_elements,
                     snapshot.occupations, fixture.handles.stream()));
      CHECK(download(state.density.density, state.density.density_elements, snapshot.density,
                     fixture.handles.stream()));
      CHECK(download(state.density.energy_weighted_density, state.density.weighted_density_elements,
                     snapshot.weighted_density, fixture.handles.stream()));
      CHECK(download(state.raw_population.qsh, state.raw_population.qsh_elements, snapshot.qsh,
                     fixture.handles.stream()));
      CHECK(download(state.raw_population.qat, state.raw_population.qat_elements, snapshot.qat,
                     fixture.handles.stream()));
      CHECK(download(state.raw_population.dipole, state.raw_population.dipole_elements,
                     snapshot.dipoles, fixture.handles.stream()));
      CHECK(download(state.raw_population.quadrupole, state.raw_population.quadrupole_elements,
                     snapshot.quadrupoles, fixture.handles.stream()));
      CHECK(download(state.scc.iterations, state.scc.batch_elements, snapshot.iterations,
                     fixture.handles.stream()));
      CHECK(download(state.scc.system_statuses, state.scc.batch_elements, snapshot.statuses,
                     fixture.handles.stream()));
      CHECK(download(state.scc.converged, state.scc.batch_elements, snapshot.converged,
                     fixture.handles.stream()));
      CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
      return 0;
    };
    const auto compare_numeric = [](const char* field, const std::vector<double>& first,
                                    const std::vector<double>& second) {
      if (first.size() != second.size()) {
        std::fprintf(stderr, "%s extent mismatch: %zu != %zu\n", field, first.size(),
                     second.size());
        return false;
      }
      for (std::size_t index = 0; index < first.size(); ++index) {
        if ((std::isnan(first[index]) && std::isnan(second[index])) ||
            near(first[index], second[index], 1.0e-10)) {
          continue;
        }
        std::fprintf(stderr, "%s mismatch at %zu: chain=%.17g monolithic=%.17g\n", field, index,
                     first[index], second[index]);
        return false;
      }
      return true;
    };
    const auto compare_states = [&](const BenchmarkState& chain_state,
                                    const BenchmarkState& monolithic_state) -> int {
      CHECK(compare_numeric("benchmark free energies", chain_state.free_energies,
                            monolithic_state.free_energies));
      CHECK(compare_numeric("benchmark eigenvalues", chain_state.eigenvalues,
                            monolithic_state.eigenvalues));
      CHECK(compare_numeric("benchmark occupations", chain_state.occupations,
                            monolithic_state.occupations));
      CHECK(compare_numeric("benchmark density", chain_state.density, monolithic_state.density));
      CHECK(compare_numeric("benchmark weighted density", chain_state.weighted_density,
                            monolithic_state.weighted_density));
      CHECK(compare_numeric("benchmark qsh", chain_state.qsh, monolithic_state.qsh));
      CHECK(compare_numeric("benchmark qat", chain_state.qat, monolithic_state.qat));
      CHECK(compare_numeric("benchmark dipoles", chain_state.dipoles, monolithic_state.dipoles));
      CHECK(compare_numeric("benchmark quadrupoles", chain_state.quadrupoles,
                            monolithic_state.quadrupoles));
      CHECK(chain_state.iterations == monolithic_state.iterations);
      CHECK(chain_state.statuses == monolithic_state.statuses);
      CHECK(chain_state.converged == monolithic_state.converged);
      return 0;
    };

    for (const std::int64_t active_denominator : {1, 2, 4}) {
      const std::int64_t active_every = active_denominator;
      std::vector<std::uint64_t> terminal_iterations(static_cast<std::size_t>(batch_size));
      std::int64_t active_members = 0;
      for (std::int64_t system = 0; system < batch_size; ++system) {
        const bool terminal = system % active_every != 0;
        /* Terminal members start at the maximum so the first activity
         * derivation excludes them from every subsequent body. */
        terminal_iterations[static_cast<std::size_t>(system)] = terminal ? maximum : 0u;
        if (!terminal) {
          ++active_members;
        }
      }
      /* Batch 1 has one member that stays active for every denominator, so
       * those tiers are identical to the full-activity tier; only record the
       * first and keep the loop deterministic. */
      if (active_members == batch_size && active_denominator != 1) {
        continue;
      }
      const auto reset_ladder = [&]() -> bool {
        Gfn2SccIterationInitializationReady ready{};
        if (!fixture.initializer
                 .upload_async(fixture.iteration_arena.get(), fixture.iteration_arena.bytes(),
                               ready, fixture.handles.stream())
                 .success()) {
          return false;
        }
        return cudaMemcpyAsync(fixture.binding.state.scc.iterations, terminal_iterations.data(),
                               terminal_iterations.size() * sizeof(std::uint64_t),
                               cudaMemcpyHostToDevice, fixture.handles.stream()) == cudaSuccess;
      };

      const auto measure = [&](const char* mode, const Gfn2SccLoopCudaGraphOwner& owner,
                               std::uint64_t& bodies) -> int {
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(kSamples));
        double total_ms = 0.0;
        float minimum_ms = std::numeric_limits<float>::infinity();
        for (int sample = -kWarmup; sample < kSamples; ++sample) {
          CHECK(reset_ladder());
          CUDA_CHECK(cudaEventRecord(start, fixture.handles.stream()));
          const Gfn2SccLoopLaunchResult launch = owner.launch(fixture.handles.stream());
          CHECK(launch.success());
          CUDA_CHECK(cudaEventRecord(stop, fixture.handles.stream()));
          CUDA_CHECK(cudaEventSynchronize(stop));
          float elapsed_ms = 0.0F;
          CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
          if (sample >= 0) {
            samples.push_back(elapsed_ms);
            total_ms += elapsed_ms;
            minimum_ms = std::min(minimum_ms, elapsed_ms);
          }
          std::uint32_t terminal_active = 1u;
          CHECK(download_value(owner.canonical_active_count_device(), terminal_active,
                               fixture.handles.stream()));
          CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
          CHECK(terminal_active == 0u);
        }
        CHECK(
            download_value(owner.numerical_body_count_device(), bodies, fixture.handles.stream()));
        CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
        std::sort(samples.begin(), samples.end());
        const double median_ms = samples[samples.size() / 2u];
        const std::size_t p95_index =
            static_cast<std::size_t>((static_cast<double>(samples.size()) - 1.0) * 0.95);
        std::printf(
            "{\"record_type\":\"measurement\",\"batch\":%lld,\"active_fraction\":%g,"
            "\"mode\":\"%s\",\"mean_ms\":%.6f,\"min_ms\":%.6f,\"median_ms\":%.6f,"
            "\"p95_ms\":%.6f,\"samples\":%d,\"numerical_body_count\":%llu,"
            "\"raw_samples_ms\":[",
            static_cast<long long>(batch_size), 1.0 / static_cast<double>(active_every), mode,
            total_ms / static_cast<double>(kSamples), static_cast<double>(minimum_ms), median_ms,
            samples[p95_index], kSamples, static_cast<unsigned long long>(bodies));
        for (std::size_t index = 0; index < samples.size(); ++index) {
          std::printf(index == 0 ? "%.6f" : ",%.6f", samples[index]);
        }
        std::printf("]}\n");
        return 0;
      };

      /* Chains and monolithic must agree on the number of numerical bodies for
       * the identical ladder. One extra reset restores the shared state so the
       * parity check below has a clean final state. */
      std::uint64_t chain_bodies = 0u;
      CHECK(measure("dispatch_chain", chain, chain_bodies) == 0);
      BenchmarkState chain_state;
      CHECK(download_state(chain_state) == 0);
      std::uint64_t monolithic_bodies = 0u;
      CHECK(measure("monolithic_tail", monolithic, monolithic_bodies) == 0);
      BenchmarkState monolithic_state;
      CHECK(download_state(monolithic_state) == 0);
      CHECK(chain_bodies == monolithic_bodies);
      CHECK(compare_states(chain_state, monolithic_state) == 0);
      const auto verify_ledger = [&]() -> int {
        /* The chain and monolithic runs share one binding, so their public
         * state memory is identical after each terminal run. Verify the ledger
         * this terminal state must satisfy under the seeded ladder: terminal
         * members keep their seeded iteration count (published untouched), and
         * active members advance by exactly the observed body count while only
         * their status/converged flags stay consistent. */
        for (std::int64_t system = 0; system < batch_size; ++system) {
          const bool terminal = terminal_iterations[static_cast<std::size_t>(system)] != 0u;
          if (terminal) {
            CHECK(monolithic_state.iterations[static_cast<std::size_t>(system)] == maximum);
          } else {
            /* Active members may converge at different rates, but none may
             * exceed the observed body count and every active member must have
             * participated. */
            CHECK(monolithic_state.iterations[static_cast<std::size_t>(system)] >= 1u);
            CHECK(monolithic_state.iterations[static_cast<std::size_t>(system)] <= chain_bodies);
          }
          CHECK(monolithic_state.statuses[static_cast<std::size_t>(system)] ==
                    XTBLOOM_STATUS_SUCCESS ||
                monolithic_state.statuses[static_cast<std::size_t>(system)] ==
                    XTBLOOM_STATUS_SCC_NOT_CONVERGED);
          CHECK(monolithic_state.converged[static_cast<std::size_t>(system)] <= 1u);
        }
        return 0;
      };
      CHECK(verify_ledger() == 0);
      /* CPU parity is meaningful only when every peer is active: terminal
       * peers legitimately publish untouched NaN/flags that the NaN-tolerant
       * full-state comparison cannot reconcile. For the all-active ladder the
       * chain must match the CPU sequential reference exactly. */
      if (active_every == 1) {
        std::string restore_error;
        CHECK(fixture.host.restore(initial, restore_error) == XTBLOOM_STATUS_SUCCESS);
        std::uint64_t reference_bodies = 0u;
        CHECK(run_host_until_globally_terminal(fixture.host, reference_bodies) == 0);
        CHECK(reference_bodies == chain_bodies);
        CHECK(compare_graph_loop_cpu_parity(fixture.host, fixture.binding,
                                            fixture.handles.stream()) == 0);
        CHECK(fixture.host.restore(initial, restore_error) == XTBLOOM_STATUS_SUCCESS);
      }
    }

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
  }
  return 0;
}

int benchmark_conditional_graph_vs_bounded_fallback() {
  constexpr int kWarmup = 3;
  constexpr int kSamples = 20;
  std::puts("batch,mode,mean_ms,min_ms,numerical_body_count");
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    ProductionFixture fixture;
    CHECK(fixture.create(false, batch_size));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

    Gfn2SccLoopCudaGraphOwner conditional;
    CHECK(conditional.build(fixture.binding).conditional_graph_ready());
    Gfn2SccIterationDevicePlan fallback_plan = fixture.binding.plan;
    fallback_plan.eigensolver_provider.capture_mode =
        Gfn2SccIterationProviderCaptureMode::kUncapturedSegmentRequired;
    Gfn2SccIterationBinding fallback_binding{};
    CHECK(bind_gfn2_scc_iteration_cuda(fallback_plan, fixture.binding.input, fixture.binding.state,
                                       fixture.binding.workspace, fallback_binding)
              .error == Gfn2SccIterationBindingError::kSuccess);
    Gfn2SccLoopCudaGraphOwner fallback;
    CHECK(fallback.build(fallback_binding).success());
    CHECK(!fallback.conditional_graph_ready());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const std::vector<std::uint64_t> penultimate(static_cast<std::size_t>(batch_size),
                                                 fixture.host.options().maximum_iterations - 1u);

    const auto reset_one_remaining = [&]() -> bool {
      Gfn2SccIterationInitializationReady ready{};
      if (!fixture.initializer
               .upload_async(fixture.iteration_arena.get(), fixture.iteration_arena.bytes(), ready,
                             fixture.handles.stream())
               .success()) {
        return false;
      }
      return cudaMemcpyAsync(fixture.binding.state.scc.iterations, penultimate.data(),
                             penultimate.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                             fixture.handles.stream()) == cudaSuccess;
    };
    const auto measure = [&](const char* mode, const Gfn2SccLoopCudaGraphOwner& owner,
                             std::uint64_t expected_bodies) -> int {
      double total_ms = 0.0;
      float minimum_ms = std::numeric_limits<float>::infinity();
      for (int sample = -kWarmup; sample < kSamples; ++sample) {
        CHECK(reset_one_remaining());
        CUDA_CHECK(cudaEventRecord(start, fixture.handles.stream()));
        const Gfn2SccLoopLaunchResult launch = owner.launch(fixture.handles.stream());
        CHECK(launch.success());
        CUDA_CHECK(cudaEventRecord(stop, fixture.handles.stream()));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        if (sample >= 0) {
          total_ms += elapsed_ms;
          minimum_ms = std::min(minimum_ms, elapsed_ms);
        }
      }
      std::printf("%lld,%s,%.6f,%.6f,%llu\n", static_cast<long long>(batch_size), mode,
                  total_ms / static_cast<double>(kSamples), static_cast<double>(minimum_ms),
                  static_cast<unsigned long long>(expected_bodies));
      return 0;
    };

    CHECK(measure("conditional", conditional, 1u) == 0);
    std::uint64_t conditional_bodies = 0u;
    CHECK(download_value(conditional.numerical_body_count_device(), conditional_bodies,
                         fixture.handles.stream()));
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
    CHECK(conditional_bodies == 1u);
    CHECK(measure("bounded_fallback", fallback, fixture.host.options().maximum_iterations) == 0);
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
  }
  return 0;
}

std::vector<double> changed_core_hamiltonian(const HostSccCase& host) {
  std::vector<double> changed = host.h0();
  const auto& layout = host.wavefunction_layout();
  const auto& matrix_offsets = host.h0_plan().matrix_offsets;
  for (std::int64_t system = 0; system < layout.batch_size; ++system) {
    const std::int64_t n =
        layout.batch_orbital_offsets[system + 1] - layout.batch_orbital_offsets[system];
    /* H0 has one physical matrix per system even when the wavefunction owns
     * separate alpha/beta coefficient matrices. Use the H0 plan offsets so a
     * changed-input replay never mistakes spin-expanded state for input. */
    const std::int64_t matrix_begin = matrix_offsets[static_cast<std::size_t>(system)];
    for (std::int64_t row = 0; row < n; ++row) {
      for (std::int64_t column = row; column < n; ++column) {
        const double shift = 2.5e-4 * static_cast<double>(system + 1) +
                             5.0e-5 * static_cast<double>(row + column + 1);
        changed[static_cast<std::size_t>(matrix_begin + row * n + column)] += shift;
        if (row != column) {
          changed[static_cast<std::size_t>(matrix_begin + column * n + row)] += shift;
        }
      }
    }
  }
  return changed;
}

int test_production_graph_changed_input_replay(std::int64_t batch_size,
                                               bool mixed_spin_batch = false,
                                               bool optional_components = false) {
  ProductionFixture fixture;
  CHECK(fixture.create(optional_components, batch_size, false, mixed_spin_batch));
  CHECK(fixture.binding.plan.eigensolver_provider.capture_mode ==
        Gfn2SccIterationProviderCaptureMode::kGraphSupported);
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  const HostSccCheckpoint initial = fixture.host.checkpoint();
  const void* const stable_iteration_arena = fixture.iteration_arena.get();
  const void* const stable_input_arena = fixture.input_arena.get();
  const void* const stable_setup_arena = fixture.eigensolver_setup_arena.get();

  GraphResources graph;
  CUDA_CHECK(cudaStreamBeginCapture(fixture.handles.stream(), cudaStreamCaptureModeThreadLocal));
  const Gfn2SccLoopLaunchResult captured =
      launch_gfn2_restricted_scc_loop_cuda(fixture.binding, fixture.handles.stream());
  CHECK(captured.success());
  CHECK(captured.submitted_iterations == fixture.host.options().maximum_iterations);
  CUDA_CHECK(cudaStreamEndCapture(fixture.handles.stream(), graph.graph_address()));
  CHECK(graph.graph() != nullptr);
  CUDA_CHECK(cudaGraphInstantiate(graph.executable_address(), graph.graph(), 0));

  CUDA_CHECK(cudaGraphLaunch(graph.executable(), fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  CHECK(run_host_fixed_scc_loop(fixture.host) == 0);
  CHECK(compare_graph_loop_cpu_parity(fixture.host, fixture.binding, fixture.handles.stream()) ==
        0);

  std::vector<double> first_free_energies;
  CHECK(download(fixture.binding.state.scc.free_energies, fixture.binding.state.scc.batch_elements,
                 first_free_energies, fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  std::string error;
  CHECK(fixture.host.restore(initial, error) == XTBLOOM_STATUS_SUCCESS);
  const std::vector<double> changed_h0 = changed_core_hamiltonian(fixture.host);
  fixture.host.h0() = changed_h0;
  Gfn2SccIterationInitializationReady ready{};
  CHECK(fixture.initializer
            .upload_async(fixture.iteration_arena.get(), fixture.iteration_arena.bytes(), ready,
                          fixture.handles.stream())
            .success());
  CUDA_CHECK(cudaMemcpyAsync(const_cast<double*>(fixture.binding.input.hamiltonian.h0),
                             changed_h0.data(), changed_h0.size() * sizeof(double),
                             cudaMemcpyHostToDevice, fixture.handles.stream()));
  CUDA_CHECK(cudaGraphLaunch(graph.executable(), fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  CHECK(run_host_fixed_scc_loop(fixture.host) == 0);
  CHECK(compare_graph_loop_cpu_parity(fixture.host, fixture.binding, fixture.handles.stream()) ==
        0);

  std::vector<double> changed_free_energies;
  CHECK(download(fixture.binding.state.scc.free_energies, fixture.binding.state.scc.batch_elements,
                 changed_free_energies, fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  CHECK(first_free_energies.size() == changed_free_energies.size());
  bool numerical_input_was_consumed = false;
  for (std::size_t index = 0; index < first_free_energies.size(); ++index) {
    numerical_input_was_consumed =
        numerical_input_was_consumed ||
        !near(first_free_energies[index], changed_free_energies[index], 1.0e-10);
  }
  CHECK(numerical_input_was_consumed);
  CHECK(fixture.iteration_arena.get() == stable_iteration_arena);
  CHECK(fixture.input_arena.get() == stable_input_arena);
  CHECK(fixture.eigensolver_setup_arena.get() == stable_setup_arena);
  return 0;
}

int test_production_graph_device_epoch_replay() {
  constexpr std::int64_t batch_size = 8;
  constexpr std::uint64_t next_generation = kGeometryGeneration + 1u;
  ProductionFixture fixture;
  CHECK(fixture.create(false, batch_size));
  CHECK(fixture.binding.plan.eigensolver_provider.capture_mode ==
        Gfn2SccIterationProviderCaptureMode::kGraphSupported);

  std::vector<Gfn2SccCacheProvenanceBinding> provenance;
  CHECK(download(fixture.binding.plan.provenance.cache_bindings,
                 fixture.binding.plan.provenance.cache_binding_count, provenance,
                 fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  const std::uint64_t* const committed = fixture.binding.plan.geometry_cache.geometry_generations;
  for (auto& record : provenance) {
    record.provenance.generation_scope = Gfn2GenerationScope::kPerSystem;
    record.provenance.geometry_generation = 0u;
    record.provenance.system_generation_count = batch_size;
    record.provenance.system_geometry_generations = committed;
  }
  CUDA_CHECK(cudaMemcpyAsync(
      const_cast<Gfn2SccCacheProvenanceBinding*>(fixture.binding.plan.provenance.cache_bindings),
      provenance.data(), provenance.size() * sizeof(Gfn2SccCacheProvenanceBinding),
      cudaMemcpyHostToDevice, fixture.handles.stream()));

  DeviceAllocation epoch_storage;
  DeviceAllocation eligible_storage;
  CHECK(epoch_storage.allocate(sizeof(std::uint64_t)));
  CHECK(eligible_storage.allocate(static_cast<std::size_t>(batch_size) * sizeof(std::uint8_t)));
  auto* const epoch = static_cast<std::uint64_t*>(epoch_storage.get());
  auto* const eligible = static_cast<std::uint8_t*>(eligible_storage.get());
  std::vector<std::uint64_t> generations(static_cast<std::size_t>(batch_size), kGeometryGeneration);
  std::vector<std::uint8_t> eligibility(static_cast<std::size_t>(batch_size), 1u);
  CUDA_CHECK(cudaMemcpyAsync(epoch, &generations[0], sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                             fixture.handles.stream()));
  CUDA_CHECK(cudaMemcpyAsync(const_cast<std::uint64_t*>(committed), generations.data(),
                             generations.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                             fixture.handles.stream()));
  CUDA_CHECK(cudaMemcpyAsync(eligible, eligibility.data(), eligibility.size(),
                             cudaMemcpyHostToDevice, fixture.handles.stream()));
  const Gfn2GeometryEpochConsumerDevice consumer{
      {epoch, 1, kPlanToken}, committed, eligible, batch_size, kPlanToken};

  GraphResources graph;
  CUDA_CHECK(cudaStreamBeginCapture(fixture.handles.stream(), cudaStreamCaptureModeThreadLocal));
  const Gfn2SccIterationLaunchResult captured = launch_gfn2_restricted_scc_iteration_cuda(
      fixture.binding, consumer, fixture.handles.stream());
  CHECK(captured.success());
  CUDA_CHECK(cudaStreamEndCapture(fixture.handles.stream(), graph.graph_address()));
  CUDA_CHECK(cudaGraphInstantiate(graph.executable_address(), graph.graph(), nullptr, nullptr, 0u));

  CUDA_CHECK(cudaGraphLaunch(graph.executable(), fixture.handles.stream()));
  std::vector<std::uint64_t> iterations;
  CHECK(download(fixture.binding.state.scc.iterations, batch_size, iterations,
                 fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  CHECK(std::all_of(iterations.begin(), iterations.end(),
                    [](std::uint64_t value) { return value == 1u; }));

  Gfn2SccIterationInitializationReady ready{};
  CHECK(fixture.initializer
            .upload_async(fixture.iteration_arena.get(), fixture.iteration_arena.bytes(), ready,
                          fixture.handles.stream())
            .success());
  generations.assign(static_cast<std::size_t>(batch_size), next_generation);
  generations[1] = kGeometryGeneration;
  eligibility[2] = 0u;
  CUDA_CHECK(cudaMemcpyAsync(epoch, &next_generation, sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                             fixture.handles.stream()));
  CUDA_CHECK(cudaMemcpyAsync(const_cast<std::uint64_t*>(committed), generations.data(),
                             generations.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                             fixture.handles.stream()));
  std::vector<std::uint64_t> factor_generations(static_cast<std::size_t>(batch_size),
                                                next_generation);
  CUDA_CHECK(cudaMemcpyAsync(fixture.binding.plan.overlap_cache.geometry_generations,
                             factor_generations.data(),
                             factor_generations.size() * sizeof(std::uint64_t),
                             cudaMemcpyHostToDevice, fixture.handles.stream()));
  CUDA_CHECK(cudaMemcpyAsync(eligible, eligibility.data(), eligibility.size(),
                             cudaMemcpyHostToDevice, fixture.handles.stream()));
  CUDA_CHECK(cudaGraphLaunch(graph.executable(), fixture.handles.stream()));
  std::vector<xtbloom_status_t> pending_statuses;
  std::vector<std::uint64_t> failures;
  CHECK(download(fixture.binding.state.scc.iterations, batch_size, iterations,
                 fixture.handles.stream()));
  CHECK(download(fixture.binding.workspace.ledger.pending_statuses, batch_size, pending_statuses,
                 fixture.handles.stream()));
  CHECK(download(fixture.binding.workspace.ledger.system_failure_records, batch_size, failures,
                 fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  CHECK(iterations[0] == 1u);
  CHECK(iterations[1] == 0u);
  CHECK(iterations[2] == 0u);
  for (const std::size_t system : {1u, 2u}) {
    CHECK(pending_statuses[system] == XTBLOOM_STATUS_INTERNAL_ERROR);
    CHECK(failures[system] ==
          gfn2_scc_stage_failure_record(
              Gfn2SccStageId::kGeometry,
              static_cast<std::uint32_t>(Gfn2SccIterationControlCode::kStaleGeneration)));
  }

  CHECK(fixture.initializer
            .upload_async(fixture.iteration_arena.get(), fixture.iteration_arena.bytes(), ready,
                          fixture.handles.stream())
            .success());
  generations.assign(static_cast<std::size_t>(batch_size), next_generation);
  eligibility.assign(static_cast<std::size_t>(batch_size), 1u);
  CUDA_CHECK(cudaMemcpyAsync(const_cast<std::uint64_t*>(committed), generations.data(),
                             generations.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                             fixture.handles.stream()));
  CUDA_CHECK(cudaMemcpyAsync(eligible, eligibility.data(), eligibility.size(),
                             cudaMemcpyHostToDevice, fixture.handles.stream()));
  CUDA_CHECK(cudaGraphLaunch(graph.executable(), fixture.handles.stream()));
  CHECK(download(fixture.binding.state.scc.iterations, batch_size, iterations,
                 fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
  CHECK(std::all_of(iterations.begin(), iterations.end(),
                    [](std::uint64_t value) { return value == 1u; }));
  return 0;
}

int test_production_loop_cpu_parity(std::int64_t batch_size, bool optional_components,
                                    bool use_default_stream, bool use_ordered_stream = false,
                                    std::uint64_t resumed_iterations = 0u,
                                    bool mixed_spin_batch = false,
                                    const std::vector<SmallSystemKind>& systems = {},
                                    double electronic_temperature = 0.0) {
  ProductionFixture fixture;
  CHECK(fixture.create(optional_components, batch_size, false, mixed_spin_batch, systems,
                       electronic_temperature));
  CHECK(fixture.host.batch_size() == batch_size);
  CHECK(!(use_default_stream && use_ordered_stream));

  OrderedExecutionStream ordered_stream;
  cudaStream_t execution_stream = fixture.handles.stream();
  if (use_ordered_stream) {
    CHECK(ordered_stream.create_after(fixture.handles.stream()));
    execution_stream = ordered_stream.stream();
  } else if (use_default_stream) {
    /* Setup uploads are ordered on the owner's nonblocking stream. A public
     * caller using another stream must establish the same one-time dependency
     * before entering the allocation-free hot loop. */
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
    execution_stream = nullptr;
  }

  std::string error;
  for (std::uint64_t iteration = 0u; iteration < resumed_iterations; ++iteration) {
    const Gfn2SccIterationLaunchResult resumed_launch =
        launch_gfn2_restricted_scc_iteration_cuda(fixture.binding, execution_stream);
    CHECK(resumed_launch.success());
    const xtbloom_status_t cpu_status = fixture.host.run_one_iteration(error);
    CHECK(cpu_status == XTBLOOM_STATUS_SUCCESS || cpu_status == XTBLOOM_STATUS_SCC_NOT_CONVERGED ||
          cpu_status == XTBLOOM_STATUS_EIGENSOLVER_FAILED);
  }

  const Gfn2SccLoopLaunchResult launch =
      launch_gfn2_restricted_scc_loop_cuda(fixture.binding, execution_stream);
  if (!launch.success()) {
    std::fprintf(stderr,
                 "production SCC loop launch failed: submitted=%llu status=%u stage=%u "
                 "binding_error=%u binding_field=%u cuda=%d cublas=%d cusolver=%d\n",
                 static_cast<unsigned long long>(launch.submitted_iterations),
                 static_cast<unsigned>(launch.iteration.status),
                 static_cast<unsigned>(launch.iteration.stage),
                 static_cast<unsigned>(launch.iteration.binding.error),
                 static_cast<unsigned>(launch.iteration.binding.field),
                 static_cast<int>(launch.iteration.cuda_status),
                 static_cast<int>(launch.iteration.cublas_status),
                 static_cast<int>(launch.iteration.cusolver_status));
  }
  CHECK(launch.success());
  CHECK(launch.submitted_iterations == fixture.host.options().maximum_iterations);

  /* Observe the first completed loop before submitting a second one. This
   * distinguishes a genuinely terminal public-state replay from an
   * implementation where the second loop merely finishes work omitted by the
   * first submission. */
  CUDA_CHECK(cudaStreamSynchronize(execution_stream));
  std::vector<double> first_coefficients;
  std::vector<double> first_density;
  std::vector<double> first_qsh;
  std::vector<double> first_free_energies;
  std::vector<double> first_previous_free_energies;
  std::vector<double> first_free_energy_changes;
  std::vector<double> first_mixer_inputs;
  std::vector<std::uint64_t> first_iterations;
  std::vector<xtbloom_status_t> first_statuses;
  std::vector<std::uint8_t> first_converged;
  const auto& first_state = fixture.binding.state;
  CHECK(download(first_state.eigenpairs.coefficients, first_state.eigenpairs.coefficient_elements,
                 first_coefficients, fixture.handles.stream()));
  CHECK(download(first_state.density.density, first_state.density.density_elements, first_density,
                 fixture.handles.stream()));
  CHECK(download(first_state.raw_population.qsh, first_state.raw_population.qsh_elements, first_qsh,
                 fixture.handles.stream()));
  CHECK(download(first_state.scc.free_energies, first_state.scc.batch_elements, first_free_energies,
                 fixture.handles.stream()));
  CHECK(download(first_state.scc.previous_free_energies, first_state.scc.batch_elements,
                 first_previous_free_energies, fixture.handles.stream()));
  CHECK(download(first_state.scc.free_energy_changes, first_state.scc.batch_elements,
                 first_free_energy_changes, fixture.handles.stream()));
  CHECK(download(first_state.mixer.current_inputs, first_state.mixer.total_vector_elements,
                 first_mixer_inputs, fixture.handles.stream()));
  CHECK(download(first_state.scc.iterations, first_state.scc.batch_elements, first_iterations,
                 fixture.handles.stream()));
  CHECK(download(first_state.scc.system_statuses, first_state.scc.batch_elements, first_statuses,
                 fixture.handles.stream()));
  CHECK(download(first_state.scc.converged, first_state.scc.batch_elements, first_converged,
                 fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  const Gfn2SccLoopLaunchResult repeat_launch =
      launch_gfn2_restricted_scc_loop_cuda(fixture.binding, execution_stream);
  CHECK(repeat_launch.success());
  CHECK(repeat_launch.submitted_iterations == fixture.host.options().maximum_iterations);
  if (execution_stream != fixture.handles.stream()) {
    /* Downloads below use the fixture stream, so complete the independent
     * execution-stream replay first. This synchronization is test observation,
     * not part of the production loop launcher. */
    CUDA_CHECK(cudaStreamSynchronize(execution_stream));
  }

  /* The CPU reference is advanced through the same fixed iteration bound.
   * Once all peers are terminal its active predicate makes later calls public-
   * state no-ops, matching the CUDA publication gate. The CUDA provider keeps
   * a fixed bucket schedule until #80 compacts inactive members. */
  for (std::uint64_t iteration = 0u; iteration < fixture.host.options().maximum_iterations;
       ++iteration) {
    const xtbloom_status_t status = fixture.host.run_one_iteration(error);
    if (status != XTBLOOM_STATUS_SUCCESS && status != XTBLOOM_STATUS_SCC_NOT_CONVERGED &&
        status != XTBLOOM_STATUS_EIGENSOLVER_FAILED) {
      std::fprintf(stderr, "CPU production SCC loop failed at %llu: status=%d error=%s\n",
                   static_cast<unsigned long long>(iteration), status, error.c_str());
      return __LINE__;
    }
  }
  const auto& layout = fixture.host.wavefunction_layout();
  const auto& state = fixture.binding.state;
  std::vector<double> eigenvalues;
  std::vector<double> coefficients;
  std::vector<double> occupations;
  std::vector<double> density;
  std::vector<double> weighted_density;
  std::vector<double> qsh;
  std::vector<double> qat;
  std::vector<double> dipoles;
  std::vector<double> quadrupoles;
  std::vector<double> free_energies;
  std::vector<double> previous_free_energies;
  std::vector<double> free_energy_changes;
  std::vector<double> residual_rms;
  std::vector<double> internal_energies;
  std::vector<double> entropies;
  std::vector<double> current_inputs;
  std::vector<std::uint64_t> iterations;
  std::vector<xtbloom_status_t> statuses;
  std::vector<std::uint8_t> converged;

  CHECK(download(state.eigenpairs.eigenvalues, state.eigenpairs.eigenvalue_elements, eigenvalues,
                 fixture.handles.stream()));
  CHECK(download(state.eigenpairs.coefficients, state.eigenpairs.coefficient_elements, coefficients,
                 fixture.handles.stream()));
  CHECK(download(state.occupations.occupations, state.occupations.occupation_elements, occupations,
                 fixture.handles.stream()));
  CHECK(download(state.density.density, state.density.density_elements, density,
                 fixture.handles.stream()));
  CHECK(download(state.density.energy_weighted_density, state.density.weighted_density_elements,
                 weighted_density, fixture.handles.stream()));
  CHECK(download(state.raw_population.qsh, state.raw_population.qsh_elements, qsh,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.qat, state.raw_population.qat_elements, qat,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.dipole, state.raw_population.dipole_elements, dipoles,
                 fixture.handles.stream()));
  CHECK(download(state.raw_population.quadrupole, state.raw_population.quadrupole_elements,
                 quadrupoles, fixture.handles.stream()));
  CHECK(download(state.scc.free_energies, state.scc.batch_elements, free_energies,
                 fixture.handles.stream()));
  CHECK(download(state.scc.previous_free_energies, state.scc.batch_elements, previous_free_energies,
                 fixture.handles.stream()));
  CHECK(download(state.scc.free_energy_changes, state.scc.batch_elements, free_energy_changes,
                 fixture.handles.stream()));
  CHECK(download(state.scc.residual_rms, state.scc.batch_elements, residual_rms,
                 fixture.handles.stream()));
  CHECK(download(state.free_energy.internal_energy, state.free_energy.internal_energy_elements,
                 internal_energies, fixture.handles.stream()));
  CHECK(download(state.free_energy.entropy, state.free_energy.entropy_elements, entropies,
                 fixture.handles.stream()));
  CHECK(download(state.mixer.current_inputs, state.mixer.total_vector_elements, current_inputs,
                 fixture.handles.stream()));
  CHECK(download(state.scc.iterations, state.scc.batch_elements, iterations,
                 fixture.handles.stream()));
  CHECK(download(state.scc.system_statuses, state.scc.batch_elements, statuses,
                 fixture.handles.stream()));
  CHECK(
      download(state.scc.converged, state.scc.batch_elements, converged, fixture.handles.stream()));
  CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));

  CHECK(coefficients == first_coefficients);
  CHECK(density == first_density);
  CHECK(qsh == first_qsh);
  CHECK(free_energies == first_free_energies);
  CHECK(previous_free_energies == first_previous_free_energies);
  CHECK(free_energy_changes == first_free_energy_changes);
  CHECK(current_inputs == first_mixer_inputs);
  CHECK(iterations == first_iterations);
  CHECK(statuses == first_statuses);
  CHECK(converged == first_converged);

  constexpr double kLoopTolerance = 1.0e-8;
  for (std::int64_t system = 0; system < fixture.host.batch_size(); ++system) {
    if (iterations[static_cast<std::size_t>(system)] !=
            fixture.host.driver_state().iterations[system] ||
        statuses[static_cast<std::size_t>(system)] !=
            fixture.host.driver_state().system_statuses[system] ||
        converged[static_cast<std::size_t>(system)] !=
            fixture.host.driver_state().converged[system]) {
      std::fprintf(stderr,
                   "loop terminal mismatch system=%lld CUDA(iter=%llu status=%d conv=%u "
                   "free=%.17g residual=%.17g) CPU(iter=%llu status=%d conv=%u free=%.17g "
                   "residual=%.17g)\n",
                   static_cast<long long>(system),
                   static_cast<unsigned long long>(iterations[static_cast<std::size_t>(system)]),
                   statuses[static_cast<std::size_t>(system)],
                   static_cast<unsigned>(converged[static_cast<std::size_t>(system)]),
                   free_energies[static_cast<std::size_t>(system)],
                   residual_rms[static_cast<std::size_t>(system)],
                   static_cast<unsigned long long>(fixture.host.driver_state().iterations[system]),
                   fixture.host.driver_state().system_statuses[system],
                   static_cast<unsigned>(fixture.host.driver_state().converged[system]),
                   fixture.host.driver_state().free_energies[system],
                   fixture.host.mixer_state().residual_rms[system]);
    }
  }
  CHECK(compare_doubles("loop eigenvalues", eigenvalues, fixture.host.wavefunction().eigenvalues,
                        layout.eigenvalues.element_count, kLoopTolerance));
  /* The existing one-step gate compares individual coefficient columns after
   * sign alignment. Across a full trajectory, exactly or numerically
   * degenerate orbitals may undergo a valid provider-dependent subspace
   * rotation. Density and energy-weighted density below are the invariant
   * full-loop correctness observables. */
  CHECK(compare_doubles("loop occupations", occupations, fixture.host.wavefunction().occupations,
                        layout.occupations.element_count, kLoopTolerance));
  CHECK(compare_doubles("loop density", density, fixture.host.wavefunction().density,
                        layout.density.element_count, kLoopTolerance));
  CHECK(compare_doubles("loop weighted density", weighted_density,
                        fixture.host.wavefunction().energy_weighted_density,
                        layout.energy_weighted_density.element_count, kLoopTolerance));
  CHECK(compare_doubles("loop qsh", qsh, fixture.host.wavefunction().qsh, layout.qsh.element_count,
                        kLoopTolerance));
  CHECK(compare_doubles("loop qat", qat, fixture.host.wavefunction().qat, layout.qat.element_count,
                        kLoopTolerance));
  CHECK(compare_doubles("loop dipoles", dipoles, fixture.host.wavefunction().dipole,
                        layout.dipole.element_count, kLoopTolerance));
  CHECK(compare_doubles("loop quadrupoles", quadrupoles, fixture.host.wavefunction().quadrupole,
                        layout.quadrupole.element_count, kLoopTolerance));
  CHECK(compare_doubles("loop free energies", free_energies,
                        fixture.host.driver_state().free_energies, fixture.host.batch_size(),
                        kLoopTolerance));
  CHECK(compare_doubles("loop previous free energies", previous_free_energies,
                        fixture.host.driver_state().previous_free_energies,
                        fixture.host.batch_size(), kLoopTolerance));
  CHECK(compare_doubles("loop free-energy changes", free_energy_changes,
                        fixture.host.driver_state().free_energy_changes, fixture.host.batch_size(),
                        kLoopTolerance));
  CHECK(compare_doubles("loop residual RMS", residual_rms, fixture.host.mixer_state().residual_rms,
                        fixture.host.batch_size(), kLoopTolerance));
  CHECK(compare_doubles("loop internal energies", internal_energies,
                        fixture.host.driver_state().internal_energies, fixture.host.batch_size(),
                        kLoopTolerance));
  CHECK(compare_doubles("loop entropies", entropies, fixture.host.driver_state().entropies,
                        fixture.host.batch_size(), kLoopTolerance));
  CHECK(compare_doubles("loop mixer current inputs", current_inputs,
                        fixture.host.mixer_state().current_inputs,
                        fixture.host.mixer_plan().total_vector_elements(), kLoopTolerance));
  for (std::int64_t system = 0; system < fixture.host.batch_size(); ++system) {
    CHECK(iterations[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().iterations[system]);
    CHECK(statuses[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().system_statuses[system]);
    CHECK(converged[static_cast<std::size_t>(system)] ==
          fixture.host.driver_state().converged[system]);
  }
  if (batch_size == 128 && !optional_components && !use_default_stream) {
    const auto converged_count =
        std::count(converged.begin(), converged.end(), static_cast<std::uint8_t>(1u));
    const auto nonconverged_count =
        std::count(statuses.begin(), statuses.end(), XTBLOOM_STATUS_SCC_NOT_CONVERGED);
    CHECK(converged_count > 0);
    CHECK(nonconverged_count > 0);
    CHECK(converged_count + nonconverged_count == batch_size);
  }
  return 0;
}

/* Exercise allocation-free, non-Graph loop modes independently from Graph
 * tooling. The B=1 case gates the unrestricted one-peer extent; larger
 * batches alternate restricted and unrestricted systems. */
int test_mixed_spin_bounded_acceptance() {
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    const int status = test_production_loop_cpu_parity(batch_size, false, false, false, 0u, true);
    if (status != 0) {
      std::fprintf(stderr, "mixed-spin fixed-loop CPU parity failed for B=%lld\n",
                   static_cast<long long>(batch_size));
      return status;
    }
  }

  int status = test_conditional_owner_bounded_fallback_parity(true);
  if (status != 0) {
    std::fprintf(stderr, "mixed-spin bounded fallback parity failed for B=8\n");
    return status;
  }

  /* The fixed-loop matrix above already covers the fixture/provider stream.
   * These two calls gate the remaining public stream-ordering contracts. */
  status = test_production_loop_cpu_parity(8, false, true, false, 0u, true);
  if (status != 0) {
    std::fprintf(stderr, "mixed-spin default-stream parity failed for B=8\n");
    return status;
  }
  status = test_production_loop_cpu_parity(8, false, false, true, 0u, true);
  if (status != 0) {
    std::fprintf(stderr, "mixed-spin event-ordered stream parity failed for B=8\n");
    return status;
  }
  return 0;
}

int test_mixed_spin_conditional_acceptance() {
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    const int status = test_conditional_graph_exact_body_count(batch_size, true);
    if (status != 0) {
      std::fprintf(stderr, "mixed-spin conditional Graph parity failed for B=%lld\n",
                   static_cast<long long>(batch_size));
      return status;
    }
  }
  return test_conditional_graph_mixed_warm_peer_parity();
}

/* Combine the isolated paths with changed-input captured-Graph replay for the
 * complete issue acceptance entry point. */
int test_mixed_spin_multi_step_and_graph_acceptance() {
  int status = test_mixed_spin_bounded_acceptance();
  if (status != 0) {
    return status;
  }
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    status = test_production_graph_changed_input_replay(batch_size, true);
    if (status != 0) {
      std::fprintf(stderr, "mixed-spin changed-input Graph replay failed for B=%lld\n",
                   static_cast<long long>(batch_size));
      return status;
    }
  }
  return test_mixed_spin_conditional_acceptance();
}

int run_mixed_spin_acceptance() {
  const int one_step = test_mixed_spin_batch_one_step_cpu_parity();
  return one_step == 0 ? test_mixed_spin_multi_step_and_graph_acceptance() : one_step;
}

int test_loop_rejects_inconsistent_plan() {
  Gfn2SccIterationBinding binding{};
  binding.plan.plan_token = 7u;
  binding.plan.activity_policy.maximum_iterations = 1u;
  binding.plan.state_policy.maximum_iterations = 1u;
  binding.plan.publication_plan.maximum_iterations = 1u;

  binding.plan.abi_version = 0u;
  Gfn2SccLoopLaunchResult result = launch_gfn2_restricted_scc_loop_cuda(binding);
  CHECK(result.iteration.status == Gfn2SccIterationLaunchStatus::kInvalidBinding);
  CHECK(result.iteration.binding.error == Gfn2SccIterationBindingError::kInvalidAbiVersion);
  CHECK(result.submitted_iterations == 0u);

  binding.plan.abi_version = kGfn2SccIterationAbiVersion;
  binding.plan.plan_token = 0u;
  result = launch_gfn2_restricted_scc_loop_cuda(binding);
  CHECK(result.iteration.binding.error == Gfn2SccIterationBindingError::kInvalidPlanToken);
  CHECK(result.submitted_iterations == 0u);

  binding.plan.plan_token = 7u;
  binding.plan.state_policy.maximum_iterations = 2u;
  result = launch_gfn2_restricted_scc_loop_cuda(binding);
  CHECK(result.iteration.binding.error == Gfn2SccIterationBindingError::kInvalidCount);
  CHECK(result.submitted_iterations == 0u);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status == cudaErrorNoDevice || count_status == cudaErrorInsufficientDriver ||
      device_count == 0) {
    (void)cudaGetLastError();
    return 0;
  }
  CUDA_CHECK(count_status);
  CUDA_CHECK(cudaSetDevice(0));
#ifdef XTBLOOM_SCC_LOOP_BENCHMARK_ONLY
  if (argc == 1) {
    return benchmark_conditional_graph_vs_bounded_fallback();
  }
#endif
  if (argc == 2 && std::strcmp(argv[1], "--benchmark") == 0) {
    return benchmark_conditional_graph_vs_bounded_fallback();
  }
  if (argc == 2 && std::strcmp(argv[1], "--benchmark-chain") == 0) {
    return benchmark_dispatch_chain_vs_monolithic();
  }
  if (argc == 6 && std::strcmp(argv[1], "--benchmark-chain-one") == 0) {
    const std::int64_t batch_size = std::strtoll(argv[2], nullptr, 10);
    const std::int64_t active_every = std::strtoll(argv[3], nullptr, 10);
    const bool use_chain = std::strcmp(argv[4], "chain") == 0;
    const bool use_monolithic = std::strcmp(argv[4], "monolithic") == 0;
    const int profiler_samples = static_cast<int>(std::strtoll(argv[5], nullptr, 10));
    if (batch_size <= 0 || active_every <= 0 || (!use_chain && !use_monolithic) ||
        profiler_samples <= 0) {
      std::fprintf(stderr,
                   "--benchmark-chain-one requires positive batch/activity/sample counts and "
                   "mode chain|monolithic\n");
      return 2;
    }
    ProductionFixture fixture;
    if (!fixture.create(false, batch_size) ||
        fixture.binding.plan.eigensolver_provider.capture_mode !=
            Gfn2SccIterationProviderCaptureMode::kGraphSupported) {
      return 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(fixture.handles.stream()));
    Gfn2SccLoopCudaGraphOwner chain;
    if (!chain.build(fixture.binding, Gfn2SccLoopGraphPreference::kDeviceDispatchChain)
             .device_dispatch_chain_ready()) {
      return 1;
    }
    Gfn2SccLoopCudaGraphOwner monolithic;
    if (!monolithic.build(fixture.binding, Gfn2SccLoopGraphPreference::kDeviceTailGraph)
             .device_tail_graph_ready()) {
      return 1;
    }
    const Gfn2SccLoopCudaGraphOwner& owner = use_chain ? chain : monolithic;
    const std::uint64_t maximum = fixture.host.options().maximum_iterations;
    std::vector<std::uint64_t> terminal_iterations(static_cast<std::size_t>(batch_size));
    for (std::int64_t system = 0; system < batch_size; ++system) {
      terminal_iterations[static_cast<std::size_t>(system)] =
          (system % active_every != 0) ? maximum : 0u;
    }
    for (int sample = 0; sample < profiler_samples; ++sample) {
      Gfn2SccIterationInitializationReady ready{};
      if (!fixture.initializer
               .upload_async(fixture.iteration_arena.get(), fixture.iteration_arena.bytes(), ready,
                             fixture.handles.stream())
               .success()) {
        return 1;
      }
      if (cudaMemcpyAsync(fixture.binding.state.scc.iterations, terminal_iterations.data(),
                          terminal_iterations.size() * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice, fixture.handles.stream()) != cudaSuccess) {
        return 1;
      }
      const Gfn2SccLoopLaunchResult launch = owner.launch(fixture.handles.stream());
      if (!launch.success()) {
        return 1;
      }
      if (cudaStreamSynchronize(fixture.handles.stream()) != cudaSuccess) {
        return 1;
      }
    }
    return 0;
  }
  if (argc == 2 && std::strcmp(argv[1], "--unrestricted-smoke") == 0) {
    return test_unrestricted_mixed_production_iteration_smoke();
  }
  if (argc == 2 && std::strcmp(argv[1], "--unrestricted-parity") == 0) {
    return test_production_iteration_cpu_parity(false, 1, true, true);
  }
  if (argc == 2 && std::strcmp(argv[1], "--mixed-parity") == 0) {
    return test_mixed_spin_batch_one_step_cpu_parity();
  }
  if (argc == 2 && std::strcmp(argv[1], "--mixed-acceptance") == 0) {
    return run_mixed_spin_acceptance();
  }
  if (argc == 2 && std::strcmp(argv[1], "--mixed-bounded") == 0) {
    return test_mixed_spin_bounded_acceptance();
  }
  if (argc == 2 && std::strcmp(argv[1], "--finite-temperature-parity") == 0) {
    int status = test_production_iteration_finite_temperature_cpu_parity();
    return status == 0 ? test_production_loop_finite_temperature_cpu_parity() : status;
  }
  if (argc == 2 && std::strcmp(argv[1], "--dispatch-chain") == 0) {
    int status = test_dispatch_chain_forced_build_and_parity(8);
    if (status != 0) {
      return status;
    }
    status = test_dispatch_chain_preference_is_honored();
    if (status != 0) {
      return status;
    }
    status = test_dispatch_chain_mixed_spin_falls_back();
    if (status != 0) {
      return status;
    }
    status = test_kauto_dispatch_chain_regime_selection();
    if (status != 0) {
      return status;
    }
    status = test_dispatch_chain_dynamic_geometry_epoch_parity();
    if (status != 0) {
      return status;
    }
    return test_dispatch_chain_owner_whole_pipeline_capture();
  }
  if (argc == 2 && std::strcmp(argv[1], "--mixed-conditional") == 0) {
    return test_mixed_spin_conditional_acceptance();
  }
  if (argc == 2 && std::strcmp(argv[1], "--large-singleton-tridiagonal") == 0) {
    return test_large_singleton_tridiagonal_graph();
  }
  if (argc == 2 && std::strcmp(argv[1], "--large-singleton-sanitizer") == 0) {
    return test_large_singleton_tridiagonal_sanitizer_smoke();
  }
  if (argc == 2 && std::strcmp(argv[1], "--deterministic-debug") == 0) {
    return test_deterministic_debug_restricted_scc_gate();
  }
  if (argc != 1) {
    std::fprintf(stderr,
                 "usage: %s "
                 "[--benchmark|--unrestricted-smoke|--unrestricted-parity|--mixed-parity|"
                 "--mixed-acceptance|--mixed-bounded|--mixed-conditional|"
                 "--dispatch-chain|--finite-temperature-parity|--large-singleton-tridiagonal|"
                 "--large-singleton-sanitizer|--deterministic-debug]\n",
                 argv[0]);
    return 2;
  }
  int status = test_loop_rejects_inconsistent_plan();
  if (status != 0) {
    return status;
  }
  status = test_unrestricted_mixed_production_iteration_smoke();
  if (status != 0) {
    return status;
  }
  status = test_production_iteration_cpu_parity(false, 1, true, true);
  if (status != 0) {
    return status;
  }
  status = test_mixed_spin_batch_one_step_cpu_parity();
  if (status != 0) {
    return status;
  }
  status = test_production_iteration_finite_temperature_cpu_parity();
  if (status != 0) {
    return status;
  }
  status = test_production_iteration_cpu_parity(false);
  if (status != 0) {
    return status;
  }
  status = test_production_iteration_cpu_parity(true);
  if (status != 0) {
    return status;
  }
  status = test_individual_coupling_cpu_parity();
  if (status != 0) {
    return status;
  }
  status = test_changed_device_overlap_is_consumed_by_production_scc();
  if (status != 0) {
    return status;
  }
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    status = test_production_graph_changed_input_replay(batch_size);
    if (status != 0) {
      return status;
    }
  }
  status = test_production_graph_changed_input_replay(8, false, true);
  if (status != 0) {
    std::fprintf(stderr, "optional-coupled changed-input Graph replay failed\n");
    return status;
  }
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    status = test_conditional_graph_exact_body_count(batch_size);
    if (status != 0) {
      return status;
    }
  }
  status = test_conditional_graph_plan_failure_forces_exit();
  if (status != 0) {
    return status;
  }
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    status = test_dispatch_chain_forced_build_and_parity(batch_size);
    if (status != 0) {
      return status;
    }
  }
  status = test_dispatch_chain_preference_is_honored();
  if (status != 0) {
    return status;
  }
  status = test_dispatch_chain_mixed_spin_falls_back();
  if (status != 0) {
    return status;
  }
  status = test_kauto_dispatch_chain_regime_selection();
  if (status != 0) {
    return status;
  }
  status = test_dispatch_chain_dynamic_geometry_epoch_parity();
  if (status != 0) {
    return status;
  }
  status = test_dispatch_chain_owner_whole_pipeline_capture();
  if (status != 0) {
    return status;
  }

  status = test_device_tail_owner_whole_pipeline_capture();
  if (status != 0) {
    return status;
  }
  status = test_conditional_owner_bounded_fallback_parity();
  if (status != 0) {
    return status;
  }
  status = test_mixed_spin_multi_step_and_graph_acceptance();
  if (status != 0) {
    return status;
  }
  status = test_production_graph_device_epoch_replay();
  if (status != 0) {
    return status;
  }
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    status = test_production_loop_cpu_parity(batch_size, false, false);
    if (status != 0) {
      return status;
    }
  }
  for (bool use_default_stream : {false, true}) {
    status = test_production_loop_cpu_parity(8, true, use_default_stream);
    if (status != 0) {
      return status;
    }
  }
  status = test_production_loop_cpu_parity(8, false, true);
  if (status != 0) {
    return status;
  }
  status = test_production_loop_cpu_parity(8, true, false, false, 2u);
  if (status != 0) {
    return status;
  }
  status = test_production_loop_cpu_parity(8, true, false, true);
  if (status != 0) {
    return status;
  }
  status = test_production_loop_finite_temperature_cpu_parity();
  if (status != 0) {
    return status;
  }
  return 0;
}
