// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "runtime/gfn1_cpu_execution.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

#include "model/gfn2/eigensolver.hpp"

#if defined(XTBLOOM_TEST_SCIPY_PREFIXED_BLAS)
#define LAPACKE_dpotrf_work scipy_LAPACKE_dpotrf_work
#define LAPACKE_dpocon_work scipy_LAPACKE_dpocon_work
#define LAPACKE_dsyevd_work scipy_LAPACKE_dsyevd_work
#define cblas_dtrsm scipy_cblas_dtrsm
#define cblas_dgemm scipy_cblas_dgemm
#endif

extern "C" {
std::int32_t LAPACKE_dpotrf_work(std::int32_t, char, std::int32_t, double*, std::int32_t);
std::int32_t LAPACKE_dpocon_work(std::int32_t, char, std::int32_t, const double*, std::int32_t,
                                 double, double*, double*, std::int32_t*);
std::int32_t LAPACKE_dsyevd_work(std::int32_t, char, char, std::int32_t, double*, std::int32_t,
                                 double*, double*, std::int32_t, std::int32_t*, std::int32_t);
void cblas_dtrsm(int, int, int, int, int, std::int32_t, std::int32_t, double, const double*,
                 std::int32_t, double*, std::int32_t);
void cblas_dgemm(int, int, int, std::int32_t, std::int32_t, std::int32_t, double, const double*,
                 std::int32_t, const double*, std::int32_t, double, double*, std::int32_t);
}

namespace allocation_test {
std::atomic<std::size_t> new_count{0u};
std::atomic<std::size_t> malloc_count{0u};
std::atomic<bool> enabled{false};
}  // namespace allocation_test

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define XTBLOOM_GFN1_CPU_EXECUTION_TEST_ASAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) && !defined(XTBLOOM_GFN1_CPU_EXECUTION_TEST_ASAN)
#define XTBLOOM_GFN1_CPU_EXECUTION_TEST_ASAN 1
#endif

#if !defined(XTBLOOM_GFN1_CPU_EXECUTION_TEST_ASAN) && defined(__GLIBC__)
extern "C" void* __libc_malloc(std::size_t size) noexcept;

/* Include raw/provider allocation in the plan-style steady-state gate. */
extern "C" void* malloc(std::size_t size) noexcept {
  if (allocation_test::enabled.load(std::memory_order_relaxed)) {
    allocation_test::malloc_count.fetch_add(1u, std::memory_order_relaxed);
  }
  return __libc_malloc(size == 0u ? 1u : size);
}
#endif

/* Interpose C++ allocations from the hidden executor as well. */
void* operator new(std::size_t size) {
  if (allocation_test::enabled.load(std::memory_order_relaxed)) {
    allocation_test::new_count.fetch_add(1u, std::memory_order_relaxed);
  }
  if (void* pointer = std::malloc(size == 0u ? 1u : size); pointer != nullptr) return pointer;
  throw std::bad_alloc();
}

void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { ::operator delete(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { ::operator delete[](pointer); }

#define CHECK(condition)                                                                   \
  do {                                                                                     \
    if (!(condition)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      return __LINE__;                                                                     \
    }                                                                                      \
  } while (false)

namespace {

using CpuLinearAlgebraBackend = xtbloom::detail::gfn2::CpuLinearAlgebraBackend;

std::atomic<std::int32_t> failed_eigensolver_dimension{0};
std::atomic<std::size_t> injected_eigensolver_failures{0u};
std::atomic<std::size_t> backend_cleanup_calls{0u};

std::int32_t injected_dpotrf(std::int32_t layout, char uplo, std::int32_t n, double* matrix,
                             std::int32_t leading_dimension) {
  return LAPACKE_dpotrf_work(layout, uplo, n, matrix, leading_dimension);
}

std::int32_t injected_dpocon(std::int32_t layout, char uplo, std::int32_t n, const double* factor,
                             std::int32_t leading_dimension, double matrix_one_norm,
                             double* reciprocal_condition, double* work,
                             std::int32_t* integer_work) {
  return LAPACKE_dpocon_work(layout, uplo, n, factor, leading_dimension, matrix_one_norm,
                             reciprocal_condition, work, integer_work);
}

std::int32_t injected_dsyevd(std::int32_t layout, char job_vectors, char uplo, std::int32_t n,
                             double* matrix, std::int32_t leading_dimension, double* eigenvalues,
                             double* work, std::int32_t work_count, std::int32_t* integer_work,
                             std::int32_t integer_work_count) {
  if (n == failed_eigensolver_dimension.load(std::memory_order_relaxed)) {
    injected_eigensolver_failures.fetch_add(1u, std::memory_order_relaxed);
    return 1;
  }
  return LAPACKE_dsyevd_work(layout, job_vectors, uplo, n, matrix, leading_dimension, eigenvalues,
                             work, work_count, integer_work, integer_work_count);
}

void injected_dtrsm(int layout, int side, int triangle, int transpose, int diagonal,
                    std::int32_t rows, std::int32_t columns, double alpha,
                    const double* triangular_matrix, std::int32_t leading_triangular,
                    double* right_hand_side, std::int32_t leading_rhs) {
  cblas_dtrsm(layout, side, triangle, transpose, diagonal, rows, columns, alpha, triangular_matrix,
              leading_triangular, right_hand_side, leading_rhs);
}

void injected_dgemm(int layout, int transpose_left, int transpose_right, std::int32_t rows,
                    std::int32_t columns, std::int32_t inner, double alpha, const double* left,
                    std::int32_t leading_left, const double* right, std::int32_t leading_right,
                    double beta, double* result, std::int32_t leading_result) {
  cblas_dgemm(layout, transpose_left, transpose_right, rows, columns, inner, alpha, left,
              leading_left, right, leading_right, beta, result, leading_result);
}

void injected_backend_cleanup() { backend_cleanup_calls.fetch_add(1u, std::memory_order_relaxed); }

CpuLinearAlgebraBackend make_injected_backend() {
  CpuLinearAlgebraBackend backend;
  std::string error;
  if (xtbloom::detail::gfn2::make_internal_test_lp64_backend(
          &injected_dpotrf, &injected_dpocon, &injected_dsyevd, &injected_dtrsm, &injected_dgemm,
          nullptr, backend, error, &injected_backend_cleanup) != XTBLOOM_STATUS_SUCCESS) {
    std::abort();
  }
  return backend;
}

void begin_allocation_counting() {
  allocation_test::new_count.store(0u, std::memory_order_relaxed);
  allocation_test::malloc_count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_release);
}

void end_allocation_counting() { allocation_test::enabled.store(false, std::memory_order_release); }

bool no_counted_allocations() {
  if (allocation_test::new_count.load(std::memory_order_relaxed) != 0u) return false;
#if !defined(XTBLOOM_GFN1_CPU_EXECUTION_TEST_ASAN) && defined(__GLIBC__)
  return allocation_test::malloc_count.load(std::memory_order_relaxed) == 0u;
#else
  return true;
#endif
}

template <typename T>
xtbloom_const_buffer_t input_buffer(const std::vector<T>& values) {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), XTBLOOM_MEMORY_HOST,
          0u};
}

template <typename T>
xtbloom_buffer_t output_buffer(std::vector<T>& values) {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), XTBLOOM_MEMORY_HOST,
          0u};
}

struct Request {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> charges;
  std::vector<std::int32_t> unpaired;
  std::vector<std::int32_t> spin_channels;
  std::vector<std::int64_t> point_offsets;
  std::vector<double> point_positions;
  std::vector<double> point_charges;
  std::vector<double> point_gammas;
  std::vector<double> shifts;
  std::vector<std::int64_t> response_offsets;
  std::vector<double> response;
  xtbloom_batch_t batch{};
  xtbloom_compute_options_t options{};

  void bind(std::uint32_t flags) {
    batch = {};
    batch.struct_size = XTBLOOM_BATCH_V3_SIZE;
    batch.api_version = XTBLOOM_API_VERSION;
    batch.batch_size = static_cast<std::int64_t>(atom_offsets.size() - 1u);
    batch.total_atoms = static_cast<std::int64_t>(atomic_numbers.size());
    batch.total_point_charges = static_cast<std::int64_t>(point_charges.size());
    batch.total_charge_response_elements = static_cast<std::int64_t>(response.size());
    batch.atom_offsets = input_buffer(atom_offsets);
    batch.atomic_numbers = input_buffer(atomic_numbers);
    batch.positions = input_buffer(positions);
    batch.molecular_charges = input_buffer(charges);
    batch.unpaired_electrons = input_buffer(unpaired);
    batch.spin_channels = input_buffer(spin_channels);
    batch.point_charge_offsets = input_buffer(point_offsets);
    batch.point_charge_positions = input_buffer(point_positions);
    batch.point_charge_values = input_buffer(point_charges);
    batch.point_charge_gammas = input_buffer(point_gammas);
    batch.atomic_potential_shifts = input_buffer(shifts);
    batch.charge_response_offsets = input_buffer(response_offsets);
    batch.charge_response_matrix = input_buffer(response);

    options = {};
    options.struct_size = XTBLOOM_COMPUTE_OPTIONS_V3_SIZE;
    options.api_version = XTBLOOM_API_VERSION;
    options.model = XTBLOOM_MODEL_GFN1_XTB;
    options.flags = flags;
    options.max_scc_iterations = 250;
    options.charge_tolerance = 1.0e-7;
    options.energy_tolerance = 1.0e-9;
    options.electronic_temperature = XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE;
    options.scc_start_mode = XTBLOOM_SCC_START_FRESH;
    options.scc_mixer = XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN;
    options.scc_mixer_history = 8;
    options.scc_mixer_damping = 0.4;
    options.determinism = XTBLOOM_DETERMINISM_REPRODUCIBLE;
  }
};

struct Result {
  std::vector<double> energies;
  std::vector<double> forces;
  std::vector<double> atomic_charges;
  std::vector<double> point_forces;
  std::vector<std::int32_t> iterations;
  std::vector<std::uint8_t> converged;
  std::vector<std::int32_t> statuses;
  xtbloom_batch_result_t result{};

  void bind(const Request& request) {
    const std::size_t batch = static_cast<std::size_t>(request.batch.batch_size);
    const std::size_t atoms = static_cast<std::size_t>(request.batch.total_atoms);
    const std::size_t points = static_cast<std::size_t>(request.batch.total_point_charges);
    energies.assign((request.options.flags & XTBLOOM_COMPUTE_ENERGY) != 0u ? batch : 0u, 71.0);
    forces.assign((request.options.flags & XTBLOOM_COMPUTE_FORCES) != 0u ? 3u * atoms : 0u, 72.0);
    atomic_charges.assign(
        (request.options.flags & XTBLOOM_COMPUTE_ATOMIC_CHARGES) != 0u ? atoms : 0u, 73.0);
    point_forces.assign(
        (request.options.flags & XTBLOOM_COMPUTE_POINT_CHARGE_FORCES) != 0u ? 3u * points : 0u,
        74.0);
    iterations.assign(batch, 75);
    converged.assign(batch, 76u);
    statuses.assign(batch, 77);
    result = {};
    result.struct_size = XTBLOOM_BATCH_RESULT_V2_SIZE;
    result.api_version = XTBLOOM_API_VERSION;
    result.flags = 0x5a5a5a5au;
    result.energies = output_buffer(energies);
    result.forces = output_buffer(forces);
    result.atomic_charges = output_buffer(atomic_charges);
    result.point_charge_forces = output_buffer(point_forces);
    result.scc_iterations = output_buffer(iterations);
    result.scc_converged = output_buffer(converged);
    result.per_system_status = output_buffer(statuses);
  }
};

Request mixed_request(std::uint32_t flags) {
  Request request;
  request.atom_offsets = {0, 2, 3};
  request.atomic_numbers = {1, 1, 1};
  request.positions = {0.0, 0.0, -0.7, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0};
  request.charges = {0.0, 0.0};
  request.unpaired = {0, 1};
  request.spin_channels = {1, 2};
  request.point_offsets = {0, 1, 1};
  request.point_positions = {2.0, 0.0, 0.0};
  request.point_charges = {0.2};
  request.point_gammas = {0.6};
  request.bind(flags);
  return request;
}

bool finite_vector(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

bool near(double first, double second, double tolerance = 3.0e-8) {
  return std::isfinite(first) && std::isfinite(second) &&
         std::abs(first - second) <= tolerance * std::max({1.0, std::abs(first), std::abs(second)});
}

bool all_nan(const std::vector<double>& values, std::size_t begin, std::size_t end) {
  return std::all_of(values.begin() + static_cast<std::ptrdiff_t>(begin),
                     values.begin() + static_cast<std::ptrdiff_t>(end),
                     [](double value) { return std::isnan(value); });
}

struct ResultImage {
  explicit ResultImage(const Result& value)
      : energies(value.energies),
        forces(value.forces),
        atomic_charges(value.atomic_charges),
        point_forces(value.point_forces),
        iterations(value.iterations),
        converged(value.converged),
        statuses(value.statuses),
        flags(value.result.flags) {}

  bool matches(const Result& value) const {
    return energies == value.energies && forces == value.forces &&
           atomic_charges == value.atomic_charges && point_forces == value.point_forces &&
           iterations == value.iterations && converged == value.converged &&
           statuses == value.statuses && flags == value.result.flags;
  }

  std::vector<double> energies;
  std::vector<double> forces;
  std::vector<double> atomic_charges;
  std::vector<double> point_forces;
  std::vector<std::int32_t> iterations;
  std::vector<std::uint8_t> converged;
  std::vector<std::int32_t> statuses;
  std::uint32_t flags = 0u;
};

bool warm_rejected_atomically(xtbloom::detail::Gfn1CpuExecutionCache& cache, Request& request,
                              std::string& error) {
  request.options.scc_start_mode = XTBLOOM_SCC_START_WARM;
  Result result;
  result.bind(request);
  const ResultImage before(result);
  return xtbloom::detail::execute_gfn1_cpu(cache, request.batch, request.options, result.result,
                                           error) == XTBLOOM_STATUS_INVALID_ARGUMENT &&
         before.matches(result);
}

int test_mixed_ragged_warm_and_periodic() {
  using xtbloom::detail::execute_gfn1_cpu;
  using xtbloom::detail::Gfn1CpuExecutionCache;
  using xtbloom::detail::persistent_workspace_bytes_gfn1_cpu;
  using xtbloom::detail::prepare_gfn1_cpu;

  constexpr std::uint32_t flags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES |
                                  XTBLOOM_COMPUTE_ATOMIC_CHARGES |
                                  XTBLOOM_COMPUTE_POINT_CHARGE_FORCES;
  Request request = mixed_request(flags);
  Result result;
  result.bind(request);
  Gfn1CpuExecutionCache cache;
  std::string error;
  bool reused = true;
  CHECK(prepare_gfn1_cpu(cache, request.batch, request.options, reused, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(!reused);
  const std::size_t retained = persistent_workspace_bytes_gfn1_cpu(cache);
  CHECK(retained > 0u);
  CHECK(prepare_gfn1_cpu(cache, request.batch, request.options, reused, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(reused);
  CHECK(persistent_workspace_bytes_gfn1_cpu(cache) == retained);
  begin_allocation_counting();
  const xtbloom_status_t fresh_status =
      execute_gfn1_cpu(cache, request.batch, request.options, result.result, error);
  end_allocation_counting();
  CHECK(fresh_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(no_counted_allocations());
  CHECK(persistent_workspace_bytes_gfn1_cpu(cache) == retained);
  CHECK(result.result.flags == 0u);
  CHECK(finite_vector(result.energies) && finite_vector(result.forces) &&
        finite_vector(result.atomic_charges) && finite_vector(result.point_forces));
  CHECK(result.converged == std::vector<std::uint8_t>({1u, 1u}));
  CHECK(result.statuses ==
        std::vector<std::int32_t>({XTBLOOM_STATUS_SUCCESS, XTBLOOM_STATUS_SUCCESS}));
  CHECK(near(result.atomic_charges[0] + result.atomic_charges[1], 0.0, 2.0e-7));
  CHECK(near(result.atomic_charges[2], 0.0, 2.0e-7));

  /* Re-preparing an identical plan is setup-only and must retain the newly
   * published strict-WARM checkpoint as well as every resident allocation. */
  CHECK(prepare_gfn1_cpu(cache, request.batch, request.options, reused, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(reused);
  CHECK(persistent_workspace_bytes_gfn1_cpu(cache) == retained);

  const std::vector<double> fresh_energy = result.energies;
  request.positions[2] -= 0.02;
  request.positions[5] += 0.02;
  request.point_positions[1] += 0.01;
  request.point_charges[0] -= 0.002;
  request.point_gammas[0] += 0.01;
  request.bind(flags);
  request.options.scc_start_mode = XTBLOOM_SCC_START_WARM;
  result.bind(request);
  begin_allocation_counting();
  const xtbloom_status_t warm_status =
      execute_gfn1_cpu(cache, request.batch, request.options, result.result, error);
  end_allocation_counting();
  CHECK(warm_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(no_counted_allocations());
  CHECK(persistent_workspace_bytes_gfn1_cpu(cache) == retained);
  CHECK(finite_vector(result.energies));
  CHECK(!near(result.energies[0], fresh_energy[0], 1.0e-14));

  request.options.scc_start_mode = XTBLOOM_SCC_START_FRESH;
  Result independent;
  independent.bind(request);
  Gfn1CpuExecutionCache independent_cache;
  CHECK(execute_gfn1_cpu(independent_cache, request.batch, request.options, independent.result,
                         error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(near(result.energies[0], independent.energies[0]));
  CHECK(near(result.energies[1], independent.energies[1]));
  for (std::size_t index = 0u; index < result.forces.size(); ++index)
    CHECK(near(result.forces[index], independent.forces[index]));
  for (std::size_t index = 0u; index < result.atomic_charges.size(); ++index)
    CHECK(near(result.atomic_charges[index], independent.atomic_charges[index]));

  request.shifts = {0.01, -0.02, 0.03};
  request.response_offsets = {0, 4, 5};
  request.response = {0.02, 0.004, 0.004, -0.01, 0.03};
  request.bind(flags);
  result.bind(request);
  CHECK(execute_gfn1_cpu(cache, request.batch, request.options, result.result, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(result.result.flags == XTBLOOM_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES);
  CHECK(finite_vector(result.energies) && finite_vector(result.forces));

  /* Point and periodic structures are fixed-plan identity, while their
   * numerical values can change on a geometry-sequence WARM transaction. */
  request.shifts[0] += 0.0001;
  request.response[0] += 0.0002;
  request.bind(flags);
  request.options.scc_start_mode = XTBLOOM_SCC_START_WARM;
  result.bind(request);
  CHECK(execute_gfn1_cpu(cache, request.batch, request.options, result.result, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(result.result.flags == XTBLOOM_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES);
  return 0;
}

int test_strict_warm_and_call_failure_are_atomic() {
  using xtbloom::detail::execute_gfn1_cpu;
  using xtbloom::detail::Gfn1CpuExecutionCache;
  constexpr std::uint32_t flags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_ATOMIC_CHARGES;
  Request request = mixed_request(flags);
  request.options.scc_start_mode = XTBLOOM_SCC_START_WARM;
  Result result;
  result.bind(request);
  Gfn1CpuExecutionCache cache;
  std::string error;
  const std::vector<double> energy_before = result.energies;
  const std::vector<double> charges_before = result.atomic_charges;
  const std::vector<std::int32_t> iterations_before = result.iterations;
  const std::vector<std::uint8_t> converged_before = result.converged;
  const std::vector<std::int32_t> statuses_before = result.statuses;
  const std::uint32_t flags_before = result.result.flags;
  CHECK(execute_gfn1_cpu(cache, request.batch, request.options, result.result, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(result.energies == energy_before && result.atomic_charges == charges_before &&
        result.iterations == iterations_before && result.converged == converged_before &&
        result.statuses == statuses_before && result.result.flags == flags_before);

  request.options.scc_start_mode = XTBLOOM_SCC_START_FRESH;
  result.bind(request);
  CHECK(execute_gfn1_cpu(cache, request.batch, request.options, result.result, error) ==
        XTBLOOM_STATUS_SUCCESS);

  /* Every topology and compute-policy key is an exact strict-WARM identity.
   * Each rejected candidate leaves both caller outputs and the older compatible
   * checkpoint intact for the final control WARM below. */
  {
    Request changed = request;
    changed.atomic_numbers[0] = 2;
    changed.bind(flags);
    CHECK(warm_rejected_atomically(cache, changed, error));
  }
  {
    Request changed = request;
    changed.charges[0] = 2.0;
    changed.bind(flags);
    CHECK(warm_rejected_atomically(cache, changed, error));
  }
  {
    Request changed = request;
    changed.unpaired[0] = 2;
    changed.bind(flags);
    CHECK(warm_rejected_atomically(cache, changed, error));
  }
  {
    Request changed = request;
    changed.spin_channels[0] = 2;
    changed.bind(flags);
    CHECK(warm_rejected_atomically(cache, changed, error));
  }
  {
    Request changed = request;
    changed.point_offsets = {0, 1, 2};
    changed.point_positions.insert(changed.point_positions.end(), {1.2, -0.4, 0.3});
    changed.point_charges.push_back(-0.1);
    changed.point_gammas.push_back(0.9);
    changed.bind(flags);
    CHECK(warm_rejected_atomically(cache, changed, error));
  }
  {
    Request changed = request;
    changed.shifts = {0.01, -0.02, 0.03};
    changed.response_offsets = {0, 4, 5};
    changed.response = {0.02, 0.004, 0.004, -0.01, 0.03};
    changed.bind(flags);
    CHECK(warm_rejected_atomically(cache, changed, error));
  }
  {
    Request changed = request;
    changed.bind(flags);
    ++changed.options.max_scc_iterations;
    CHECK(warm_rejected_atomically(cache, changed, error));
  }
  {
    Request changed = request;
    changed.bind(flags);
    changed.options.charge_tolerance = std::nextafter(changed.options.charge_tolerance, 1.0);
    CHECK(warm_rejected_atomically(cache, changed, error));
  }
  {
    Request changed = request;
    changed.bind(flags);
    changed.options.energy_tolerance = std::nextafter(changed.options.energy_tolerance, 1.0);
    CHECK(warm_rejected_atomically(cache, changed, error));
  }
  {
    Request changed = request;
    changed.bind(flags);
    changed.options.electronic_temperature = 0.02;
    CHECK(warm_rejected_atomically(cache, changed, error));
  }
  {
    Request changed = request;
    changed.bind(flags);
    changed.options.scc_mixer_history = 4;
    CHECK(warm_rejected_atomically(cache, changed, error));
  }
  {
    Request changed = request;
    changed.bind(flags);
    changed.options.scc_mixer_damping = 0.2;
    CHECK(warm_rejected_atomically(cache, changed, error));
  }
  {
    Request changed = request;
    changed.bind(flags);
    changed.options.determinism = XTBLOOM_DETERMINISM_DEFAULT;
    CHECK(warm_rejected_atomically(cache, changed, error));
  }
  {
    Request changed = request;
    changed.bind(XTBLOOM_COMPUTE_ENERGY);
    CHECK(warm_rejected_atomically(cache, changed, error));
  }

  request.bind(flags);
  request.options.scc_start_mode = XTBLOOM_SCC_START_WARM;
  result.bind(request);
  CHECK(execute_gfn1_cpu(cache, request.batch, request.options, result.result, error) ==
        XTBLOOM_STATUS_SUCCESS);

  /* A call-level failure before numerical execution leaves the newly published
   * compatible checkpoint consumable. */
  Request invalid = request;
  invalid.positions[0] = std::numeric_limits<double>::quiet_NaN();
  invalid.bind(flags);
  invalid.options.scc_start_mode = XTBLOOM_SCC_START_FRESH;
  Result invalid_result;
  invalid_result.bind(invalid);
  const ResultImage invalid_before(invalid_result);
  CHECK(execute_gfn1_cpu(cache, invalid.batch, invalid.options, invalid_result.result, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(invalid_before.matches(invalid_result));
  request.bind(flags);
  request.options.scc_start_mode = XTBLOOM_SCC_START_WARM;
  result.bind(request);
  CHECK(execute_gfn1_cpu(cache, request.batch, request.options, result.result, error) ==
        XTBLOOM_STATUS_SUCCESS);
  return 0;
}

int test_malformed_charge_response_is_rejected_atomically() {
  using xtbloom::detail::execute_gfn1_cpu;
  using xtbloom::detail::Gfn1CpuExecutionCache;
  constexpr std::uint32_t flags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES;
  Request request = mixed_request(flags);
  request.shifts = {0.01, -0.02, 0.03};
  request.response_offsets = {0, 1, 5};
  request.response = {0.02, 0.004, 0.004, -0.01, 0.03};
  request.bind(flags);
  Result result;
  result.bind(request);
  const ResultImage before(result);
  Gfn1CpuExecutionCache cache;
  std::string error;
  CHECK(execute_gfn1_cpu(cache, request.batch, request.options, result.result, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(before.matches(result));
  CHECK(error.find("atoms*atoms") != std::string::npos);
  return 0;
}

int test_nonconvergence_is_data_level_and_nan_filled() {
  using xtbloom::detail::execute_gfn1_cpu;
  using xtbloom::detail::Gfn1CpuExecutionCache;
  Request request =
      mixed_request(XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES |
                    XTBLOOM_COMPUTE_ATOMIC_CHARGES | XTBLOOM_COMPUTE_POINT_CHARGE_FORCES);
  request.options.max_scc_iterations = 1;
  request.options.charge_tolerance = 1.0e-30;
  request.options.energy_tolerance = 1.0e-30;
  Result result;
  result.bind(request);
  Gfn1CpuExecutionCache cache;
  std::string error;
  CHECK(execute_gfn1_cpu(cache, request.batch, request.options, result.result, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(result.statuses[0] == XTBLOOM_STATUS_SCC_NOT_CONVERGED);
  CHECK(result.statuses[1] == XTBLOOM_STATUS_SCC_NOT_CONVERGED);
  CHECK(result.converged == std::vector<std::uint8_t>({0u, 0u}));
  CHECK(std::all_of(result.energies.begin(), result.energies.end(),
                    [](double value) { return std::isnan(value); }));
  CHECK(std::all_of(result.forces.begin(), result.forces.end(),
                    [](double value) { return std::isnan(value); }));
  CHECK(std::all_of(result.atomic_charges.begin(), result.atomic_charges.end(),
                    [](double value) { return std::isnan(value); }));
  CHECK(std::all_of(result.point_forces.begin(), result.point_forces.end(),
                    [](double value) { return std::isnan(value); }));
  return 0;
}

int test_successful_peer_survives_scc_nonconvergence() {
  using xtbloom::detail::execute_gfn1_cpu;
  using xtbloom::detail::Gfn1CpuExecutionCache;
  constexpr std::uint32_t flags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES |
                                  XTBLOOM_COMPUTE_ATOMIC_CHARGES |
                                  XTBLOOM_COMPUTE_POINT_CHARGE_FORCES;
  Request request;
  request.atom_offsets = {0, 1, 3};
  request.atomic_numbers = {2, 1, 1};
  request.positions = {0.0, 0.0, 0.0, -0.70, 0.0, 0.0, 0.70, 0.0, 0.0};
  request.charges = {0.0, 0.0};
  request.unpaired = {0, 0};
  request.spin_channels = {1, 1};
  request.point_offsets = {0, 0, 1};
  request.point_positions = {2.1, 0.3, -0.2};
  request.point_charges = {-0.1};
  request.point_gammas = {0.9};
  request.bind(flags);
  /* The closed-shell He atom converges in two iterations. The interacting H2
   * peer still has a nonzero residual at the same hard iteration ceiling. */
  request.options.max_scc_iterations = 2;

  Gfn1CpuExecutionCache cache;
  Result result;
  result.bind(request);
  std::string error;
  CHECK(execute_gfn1_cpu(cache, request.batch, request.options, result.result, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(result.statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(result.converged[0] == 1u);
  CHECK(result.iterations[0] == 2);
  CHECK(std::isfinite(result.energies[0]));
  CHECK(std::all_of(result.forces.begin(), result.forces.begin() + 3,
                    [](double value) { return std::isfinite(value); }));
  CHECK(std::isfinite(result.atomic_charges[0]));
  CHECK(result.statuses[1] == XTBLOOM_STATUS_SCC_NOT_CONVERGED);
  CHECK(result.converged[1] == 0u);
  CHECK(result.iterations[1] == 2);
  CHECK(std::isnan(result.energies[1]));
  CHECK(all_nan(result.forces, 3u, 9u));
  CHECK(all_nan(result.atomic_charges, 1u, 3u));
  CHECK(all_nan(result.point_forces, 0u, 3u));
  return 0;
}

int test_injected_eigensolver_failure_isolated_and_backend_cleaned_up() {
  using xtbloom::detail::execute_gfn1_cpu;
  using xtbloom::detail::Gfn1CpuExecutionCache;
  using xtbloom::detail::set_gfn1_cpu_linear_algebra_backend_for_testing;
  constexpr std::uint32_t flags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES |
                                  XTBLOOM_COMPUTE_ATOMIC_CHARGES |
                                  XTBLOOM_COMPUTE_POINT_CHARGE_FORCES;
  Request request;
  request.atom_offsets = {0, 1, 3};
  request.atomic_numbers = {2, 1, 1};
  request.positions = {0.0, 0.0, 0.0, -0.70, 0.0, 0.0, 0.70, 0.0, 0.0};
  request.charges = {0.0, 0.0};
  request.unpaired = {0, 0};
  request.spin_channels = {1, 1};
  request.point_offsets = {0, 0, 1};
  request.point_positions = {2.1, 0.3, -0.2};
  request.point_charges = {-0.1};
  request.point_gammas = {0.9};
  request.bind(flags);

  backend_cleanup_calls.store(0u, std::memory_order_relaxed);
  injected_eigensolver_failures.store(0u, std::memory_order_relaxed);
  const CpuLinearAlgebraBackend backend = make_injected_backend();
  failed_eigensolver_dimension.store(4, std::memory_order_relaxed);
  {
    Gfn1CpuExecutionCache cache;
    Result result;
    result.bind(request);
    std::string error;
    CHECK(set_gfn1_cpu_linear_algebra_backend_for_testing(cache, backend, error) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(execute_gfn1_cpu(cache, request.batch, request.options, result.result, error) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(injected_eigensolver_failures.load(std::memory_order_relaxed) == 1u);
    CHECK(result.statuses[0] == XTBLOOM_STATUS_SUCCESS);
    CHECK(result.converged[0] == 1u);
    CHECK(std::isfinite(result.energies[0]));
    CHECK(std::all_of(result.forces.begin(), result.forces.begin() + 3,
                      [](double value) { return std::isfinite(value); }));
    CHECK(std::isfinite(result.atomic_charges[0]));
    CHECK(result.statuses[1] == XTBLOOM_STATUS_EIGENSOLVER_FAILED);
    CHECK(result.converged[1] == 0u);
    CHECK(result.iterations[1] == 1);
    CHECK(std::isnan(result.energies[1]));
    CHECK(all_nan(result.forces, 3u, 9u));
    CHECK(all_nan(result.atomic_charges, 1u, 3u));
    CHECK(all_nan(result.point_forces, 0u, 3u));
    CHECK(backend_cleanup_calls.load(std::memory_order_relaxed) == 0u);
  }
  failed_eigensolver_dimension.store(0, std::memory_order_relaxed);
  CHECK(backend_cleanup_calls.load(std::memory_order_relaxed) == 1u);
  return 0;
}

int test_failed_peer_isolated_nan_filled_and_consumes_warm_checkpoint() {
  using xtbloom::detail::execute_gfn1_cpu;
  using xtbloom::detail::Gfn1CpuExecutionCache;
  constexpr std::uint32_t flags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES |
                                  XTBLOOM_COMPUTE_ATOMIC_CHARGES |
                                  XTBLOOM_COMPUTE_POINT_CHARGE_FORCES;
  Request request;
  request.atom_offsets = {0, 2, 4};
  request.atomic_numbers = {1, 1, 1, 1};
  request.positions = {-0.70, 0.0, 0.0, 0.70, 0.0, 0.0, 3.30, 0.0, 0.0, 4.70, 0.0, 0.0};
  request.charges = {0.0, 0.0};
  request.unpaired = {0, 0};
  request.spin_channels = {1, 1};
  request.point_offsets = {0, 0, 1};
  request.point_positions = {3.30, 1.2, -0.4};
  request.point_charges = {-0.1};
  request.point_gammas = {0.9};
  request.bind(flags);

  Gfn1CpuExecutionCache cache;
  Result result;
  result.bind(request);
  std::string error;
  CHECK(execute_gfn1_cpu(cache, request.batch, request.options, result.result, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(result.statuses ==
        std::vector<std::int32_t>({XTBLOOM_STATUS_SUCCESS, XTBLOOM_STATUS_SUCCESS}));

  /* An almost linearly dependent H2 overlap deterministically fails the
   * production overlap/eigensolver path. The healthy peer must still publish,
   * while every requested floating-point slice of the failed peer is NaN. */
  request.positions[9] = request.positions[6] + 1.1e-6;
  request.bind(flags);
  request.options.scc_start_mode = XTBLOOM_SCC_START_WARM;
  result.bind(request);
  CHECK(execute_gfn1_cpu(cache, request.batch, request.options, result.result, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(result.statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(result.converged[0] == 1u);
  CHECK(std::isfinite(result.energies[0]));
  CHECK(std::all_of(result.forces.begin(), result.forces.begin() + 6,
                    [](double value) { return std::isfinite(value); }));
  CHECK(std::all_of(result.atomic_charges.begin(), result.atomic_charges.begin() + 2,
                    [](double value) { return std::isfinite(value); }));
  CHECK(result.statuses[1] == XTBLOOM_STATUS_EIGENSOLVER_FAILED);
  CHECK(result.converged[1] == 0u);
  CHECK(result.iterations[1] == 1);
  CHECK(std::isnan(result.energies[1]));
  CHECK(all_nan(result.forces, 6u, 12u));
  CHECK(all_nan(result.atomic_charges, 2u, 4u));
  CHECK(all_nan(result.point_forces, 0u, 3u));

  /* The accepted failed predecessor consumes the earlier whole-batch
   * checkpoint. Restoring the geometry does not make strict WARM admissible. */
  request.positions[9] = 4.70;
  request.bind(flags);
  CHECK(warm_rejected_atomically(cache, request, error));

  /* Re-establish the identity, then prove an accepted failed FRESH attempt
   * consumes the checkpoint by the same predecessor rule as failed WARM. */
  request.bind(flags);
  result.bind(request);
  CHECK(execute_gfn1_cpu(cache, request.batch, request.options, result.result, error) ==
        XTBLOOM_STATUS_SUCCESS);
  request.positions[9] = request.positions[6] + 1.1e-6;
  request.bind(flags);
  result.bind(request);
  CHECK(execute_gfn1_cpu(cache, request.batch, request.options, result.result, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(result.statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(result.statuses[1] == XTBLOOM_STATUS_EIGENSOLVER_FAILED);
  request.positions[9] = 4.70;
  request.bind(flags);
  CHECK(warm_rejected_atomically(cache, request, error));
  return 0;
}

}  // namespace

int main() {
  if (const int result = test_mixed_ragged_warm_and_periodic(); result != 0) return result;
  if (const int result = test_strict_warm_and_call_failure_are_atomic(); result != 0) return result;
  if (const int result = test_malformed_charge_response_is_rejected_atomically(); result != 0)
    return result;
  if (const int result = test_nonconvergence_is_data_level_and_nan_filled(); result != 0)
    return result;
  if (const int result = test_successful_peer_survives_scc_nonconvergence(); result != 0)
    return result;
  if (const int result = test_injected_eigensolver_failure_isolated_and_backend_cleaned_up();
      result != 0) {
    return result;
  }
  return test_failed_peer_isolated_nan_filled_and_consumes_warm_checkpoint();
}
