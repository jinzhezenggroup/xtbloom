#include "model/gfn2/eigensolver.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "model/gfn2/basis.hpp"
#include "model/gfn2/occupation_binary64_policy.hpp"

#if defined(XTBLOOM_TEST_SCIPY_PREFIXED_BLAS)
#define LAPACKE_dpotrf_work scipy_LAPACKE_dpotrf_work
#define LAPACKE_dpocon_work scipy_LAPACKE_dpocon_work
#define LAPACKE_dsyevd_work scipy_LAPACKE_dsyevd_work
#define cblas_dtrsm scipy_cblas_dtrsm
#define cblas_dgemm scipy_cblas_dgemm
#endif

extern "C" {
std::int32_t LAPACKE_dpotrf_work(std::int32_t matrix_layout, char uplo, std::int32_t n,
                                 double* matrix, std::int32_t leading_dimension);
std::int32_t LAPACKE_dpocon_work(std::int32_t matrix_layout, char uplo, std::int32_t n,
                                 const double* factor, std::int32_t leading_dimension,
                                 double matrix_one_norm, double* reciprocal_condition, double* work,
                                 std::int32_t* integer_work);
std::int32_t LAPACKE_dsyevd_work(std::int32_t matrix_layout, char job_vectors, char uplo,
                                 std::int32_t n, double* matrix, std::int32_t leading_dimension,
                                 double* eigenvalues, double* work, std::int32_t work_count,
                                 std::int32_t* integer_work, std::int32_t integer_work_count);
void cblas_dtrsm(int layout, int side, int triangle, int transpose, int diagonal, std::int32_t rows,
                 std::int32_t columns, double alpha, const double* triangular_matrix,
                 std::int32_t leading_triangular, double* right_hand_side,
                 std::int32_t leading_rhs);
void cblas_dgemm(int layout, int transpose_left, int transpose_right, std::int32_t rows,
                 std::int32_t columns, std::int32_t inner, double alpha, const double* left,
                 std::int32_t leading_left, const double* right, std::int32_t leading_right,
                 double beta, double* result, std::int32_t leading_result);
}

namespace allocation_test {
std::atomic<std::size_t> new_count{0u};
std::atomic<std::size_t> malloc_count{0u};
std::atomic<bool> enabled{false};
}  // namespace allocation_test

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define XTBLOOM_EIGENSOLVER_TEST_ASAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) && !defined(XTBLOOM_EIGENSOLVER_TEST_ASAN)
#define XTBLOOM_EIGENSOLVER_TEST_ASAN 1
#endif

#if !defined(XTBLOOM_EIGENSOLVER_TEST_ASAN) && defined(__GLIBC__)
extern "C" void* __libc_malloc(std::size_t size) noexcept;

/* Count vendor-side malloc calls as well as C++ allocations on glibc builds. */
extern "C" void* malloc(std::size_t size) noexcept {
  if (allocation_test::enabled.load(std::memory_order_relaxed)) {
    allocation_test::malloc_count.fetch_add(1u, std::memory_order_relaxed);
  }
  return __libc_malloc(size == 0u ? 1u : size);
}
#endif

void* operator new(std::size_t size) {
  if (allocation_test::enabled.load(std::memory_order_relaxed)) {
    allocation_test::new_count.fetch_add(1u, std::memory_order_relaxed);
  }
  if (void* pointer = std::malloc(size == 0u ? 1u : size); pointer != nullptr) {
    return pointer;
  }
  throw std::bad_alloc();
}

void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { ::operator delete(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { ::operator delete[](pointer); }

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

using xtbloom::detail::gfn2::BasisPlan;
using xtbloom::detail::gfn2::CpuLinearAlgebraBackend;
using xtbloom::detail::gfn2::EigensolverOverlapCache;
using xtbloom::detail::gfn2::EigensolverPlan;
using xtbloom::detail::gfn2::EigensolverRecyclePolicy;
using xtbloom::detail::gfn2::EigensolverSolveMode;
using xtbloom::detail::gfn2::EigensolverSolveReport;
using xtbloom::detail::gfn2::EigensolverThermodynamicsView;
using xtbloom::detail::gfn2::EigensolverWorkspace;
using xtbloom::detail::gfn2::LapackInt;
using xtbloom::detail::gfn2::WavefunctionLayout;
using xtbloom::detail::gfn2::WavefunctionSystemView;
using xtbloom::detail::gfn2::WavefunctionView;

constexpr double kTolerance = 2.0e-11;
std::atomic<int> potrf_calls{0};
std::atomic<int> dpocon_calls{0};
std::atomic<int> dsyevd_calls{0};
std::atomic<int> dtrsm_calls{0};
std::atomic<int> dgemm_calls{0};
std::atomic<int> non_column_major_calls{0};
std::atomic<int> dpotrf_failure_call{0};
std::atomic<std::int32_t> dpotrf_failure_info{1};
std::atomic<int> dpocon_failure_call{0};
std::atomic<std::int32_t> dpocon_failure_info{1};
std::atomic<int> dsyevd_failure_call{0};
std::atomic<std::int32_t> dsyevd_failure_info{1};

std::int32_t counting_dpotrf(std::int32_t matrix_layout, char uplo, std::int32_t n, double* matrix,
                             std::int32_t leading_dimension) {
  const int call = potrf_calls.fetch_add(1, std::memory_order_relaxed) + 1;
  if (matrix_layout != 102) {
    non_column_major_calls.fetch_add(1, std::memory_order_relaxed);
  }
  if (dpotrf_failure_call.load(std::memory_order_relaxed) == call) {
    return dpotrf_failure_info.load(std::memory_order_relaxed);
  }
  return LAPACKE_dpotrf_work(matrix_layout, uplo, n, matrix, leading_dimension);
}

std::int32_t counting_dpocon(std::int32_t matrix_layout, char uplo, std::int32_t n,
                             const double* factor, std::int32_t leading_dimension,
                             double matrix_one_norm, double* reciprocal_condition, double* work,
                             std::int32_t* integer_work) {
  const int call = dpocon_calls.fetch_add(1, std::memory_order_relaxed) + 1;
  if (matrix_layout != 102) {
    non_column_major_calls.fetch_add(1, std::memory_order_relaxed);
  }
  if (dpocon_failure_call.load(std::memory_order_relaxed) == call) {
    return dpocon_failure_info.load(std::memory_order_relaxed);
  }
  return LAPACKE_dpocon_work(matrix_layout, uplo, n, factor, leading_dimension, matrix_one_norm,
                             reciprocal_condition, work, integer_work);
}

std::int32_t counting_dsyevd(std::int32_t matrix_layout, char job_vectors, char uplo,
                             std::int32_t n, double* matrix, std::int32_t leading_dimension,
                             double* eigenvalues, double* work, std::int32_t work_count,
                             std::int32_t* integer_work, std::int32_t integer_work_count) {
  const int call = dsyevd_calls.fetch_add(1, std::memory_order_relaxed) + 1;
  if (matrix_layout != 102) {
    non_column_major_calls.fetch_add(1, std::memory_order_relaxed);
  }
  if (dsyevd_failure_call.load(std::memory_order_relaxed) == call) {
    return dsyevd_failure_info.load(std::memory_order_relaxed);
  }
  return LAPACKE_dsyevd_work(matrix_layout, job_vectors, uplo, n, matrix, leading_dimension,
                             eigenvalues, work, work_count, integer_work, integer_work_count);
}

void counting_dtrsm(int layout, int side, int triangle, int transpose, int diagonal,
                    std::int32_t rows, std::int32_t columns, double alpha,
                    const double* triangular_matrix, std::int32_t leading_triangular,
                    double* right_hand_side, std::int32_t leading_rhs) {
  dtrsm_calls.fetch_add(1, std::memory_order_relaxed);
  if (layout != 102) {
    non_column_major_calls.fetch_add(1, std::memory_order_relaxed);
  }
  cblas_dtrsm(layout, side, triangle, transpose, diagonal, rows, columns, alpha, triangular_matrix,
              leading_triangular, right_hand_side, leading_rhs);
}

void counting_dgemm(int layout, int transpose_left, int transpose_right, std::int32_t rows,
                    std::int32_t columns, std::int32_t inner, double alpha, const double* left,
                    std::int32_t leading_left, const double* right, std::int32_t leading_right,
                    double beta, double* result, std::int32_t leading_result) {
  dgemm_calls.fetch_add(1, std::memory_order_relaxed);
  if (layout != 102) {
    non_column_major_calls.fetch_add(1, std::memory_order_relaxed);
  }
  cblas_dgemm(layout, transpose_left, transpose_right, rows, columns, inner, alpha, left,
              leading_left, right, leading_right, beta, result, leading_result);
}

const CpuLinearAlgebraBackend& backend() {
  static const CpuLinearAlgebraBackend created = [] {
    CpuLinearAlgebraBackend candidate;
    std::string error;
    const xtbloom_status_t status = xtbloom::detail::gfn2::make_internal_test_lp64_backend(
        &counting_dpotrf, &counting_dpocon, &counting_dsyevd, &counting_dtrsm, &counting_dgemm,
        nullptr, candidate, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      std::abort();
    }
    return candidate;
  }();
  return created;
}

bool near(double actual, double expected, double tolerance = kTolerance) {
  return std::abs(actual - expected) <= tolerance;
}

void reset_backend_spies() {
  potrf_calls.store(0, std::memory_order_relaxed);
  dpocon_calls.store(0, std::memory_order_relaxed);
  dsyevd_calls.store(0, std::memory_order_relaxed);
  dtrsm_calls.store(0, std::memory_order_relaxed);
  dgemm_calls.store(0, std::memory_order_relaxed);
  non_column_major_calls.store(0, std::memory_order_relaxed);
  dpotrf_failure_call.store(0, std::memory_order_relaxed);
  dpotrf_failure_info.store(1, std::memory_order_relaxed);
  dpocon_failure_call.store(0, std::memory_order_relaxed);
  dpocon_failure_info.store(1, std::memory_order_relaxed);
  dsyevd_failure_call.store(0, std::memory_order_relaxed);
  dsyevd_failure_info.store(1, std::memory_order_relaxed);
}

struct AlignedBuffer {
  void* data = nullptr;
  std::size_t size = 0u;

  explicit AlignedBuffer(std::size_t requested) : size(requested) {
    data = std::aligned_alloc(xtbloom::detail::gfn2::kEigensolverWorkspaceAlignment, requested);
  }

  ~AlignedBuffer() { std::free(data); }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
};

struct Evaluation {
  WavefunctionLayout layout;
  EigensolverPlan plan;
  std::unique_ptr<AlignedBuffer> wavefunction_storage;
  std::unique_ptr<AlignedBuffer> cache_storage;
  std::unique_ptr<AlignedBuffer> scratch_storage;
  WavefunctionView wavefunction;
  EigensolverOverlapCache cache;
  EigensolverWorkspace scratch;
  std::vector<xtbloom_status_t> statuses;
  std::vector<double> chemical_potentials;
  std::vector<double> entropies;
  std::vector<double> band_energies;
  std::vector<double> free_energies;

  EigensolverThermodynamicsView thermodynamics() {
    return {statuses.data(),
            statuses.size(),
            chemical_potentials.data(),
            chemical_potentials.size(),
            entropies.data(),
            entropies.size(),
            band_energies.data(),
            band_energies.size(),
            free_energies.data(),
            free_energies.size()};
  }
};

bool initialize_evaluation(const std::vector<std::int64_t>& atom_offsets,
                           const std::vector<std::int32_t>& atomic_numbers,
                           const std::vector<double>& charges,
                           const std::vector<std::int32_t>& unpaired,
                           const std::vector<std::int32_t>& spins, Evaluation& evaluation,
                           std::string& error, double minimum_rcond = 1.0e-12) {
  BasisPlan basis;
  if (xtbloom::detail::gfn2::make_basis_plan(static_cast<std::int64_t>(atom_offsets.size() - 1u),
                                             static_cast<std::int64_t>(atomic_numbers.size()),
                                             atom_offsets.data(), atomic_numbers.data(), basis,
                                             error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_wavefunction_layout(
          basis, atomic_numbers.data(), charges.data(), unpaired.data(), spins.data(),
          evaluation.layout, error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_eigensolver_plan(evaluation.layout, evaluation.plan, error,
                                                   minimum_rcond) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }

  evaluation.wavefunction_storage =
      std::make_unique<AlignedBuffer>(evaluation.layout.workspace_size_bytes);
  evaluation.cache_storage =
      std::make_unique<AlignedBuffer>(evaluation.plan.overlap_cache_size_bytes());
  evaluation.scratch_storage =
      std::make_unique<AlignedBuffer>(evaluation.plan.workspace_size_bytes());
  if (evaluation.wavefunction_storage->data == nullptr ||
      evaluation.cache_storage->data == nullptr || evaluation.scratch_storage->data == nullptr ||
      xtbloom::detail::gfn2::bind_wavefunction_view(
          evaluation.layout, evaluation.wavefunction_storage->data,
          evaluation.wavefunction_storage->size, evaluation.wavefunction,
          error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::bind_eigensolver_overlap_cache(
          evaluation.plan, evaluation.cache_storage->data, evaluation.cache_storage->size,
          evaluation.cache, error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::bind_eigensolver_workspace(
          evaluation.plan, evaluation.scratch_storage->data, evaluation.scratch_storage->size,
          evaluation.scratch, error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  const std::size_t batch = static_cast<std::size_t>(evaluation.plan.batch_size());
  evaluation.statuses.resize(batch);
  evaluation.chemical_potentials.resize(2u * batch);
  evaluation.entropies.resize(batch);
  evaluation.band_energies.resize(batch);
  evaluation.free_energies.resize(batch);
  return true;
}

void fill_outputs(Evaluation& evaluation, double sentinel) {
  std::fill_n(evaluation.wavefunction.coefficients,
              static_cast<std::size_t>(evaluation.layout.coefficients.element_count), sentinel);
  std::fill_n(evaluation.wavefunction.eigenvalues,
              static_cast<std::size_t>(evaluation.layout.eigenvalues.element_count), sentinel);
  std::fill_n(evaluation.wavefunction.occupations,
              static_cast<std::size_t>(evaluation.layout.occupations.element_count), sentinel);
  std::fill_n(evaluation.wavefunction.density,
              static_cast<std::size_t>(evaluation.layout.density.element_count), sentinel);
  std::fill_n(evaluation.wavefunction.energy_weighted_density,
              static_cast<std::size_t>(evaluation.layout.energy_weighted_density.element_count),
              sentinel);
  std::fill(evaluation.statuses.begin(), evaluation.statuses.end(),
            static_cast<xtbloom_status_t>(99));
  std::fill(evaluation.chemical_potentials.begin(), evaluation.chemical_potentials.end(), sentinel);
  std::fill(evaluation.entropies.begin(), evaluation.entropies.end(), sentinel);
  std::fill(evaluation.band_energies.begin(), evaluation.band_energies.end(), sentinel);
  std::fill(evaluation.free_energies.begin(), evaluation.free_energies.end(), sentinel);
}

bool outputs_equal_sentinel(const Evaluation& evaluation, double sentinel) {
  return std::all_of(
             evaluation.wavefunction.coefficients,
             evaluation.wavefunction.coefficients + evaluation.layout.coefficients.element_count,
             [sentinel](double value) { return value == sentinel; }) &&
         std::all_of(
             evaluation.wavefunction.eigenvalues,
             evaluation.wavefunction.eigenvalues + evaluation.layout.eigenvalues.element_count,
             [sentinel](double value) { return value == sentinel; }) &&
         std::all_of(
             evaluation.wavefunction.occupations,
             evaluation.wavefunction.occupations + evaluation.layout.occupations.element_count,
             [sentinel](double value) { return value == sentinel; }) &&
         std::all_of(evaluation.wavefunction.density,
                     evaluation.wavefunction.density + evaluation.layout.density.element_count,
                     [sentinel](double value) { return value == sentinel; }) &&
         std::all_of(evaluation.wavefunction.energy_weighted_density,
                     evaluation.wavefunction.energy_weighted_density +
                         evaluation.layout.energy_weighted_density.element_count,
                     [sentinel](double value) { return value == sentinel; }) &&
         std::all_of(evaluation.statuses.begin(), evaluation.statuses.end(),
                     [](xtbloom_status_t value) { return value == 99; }) &&
         std::all_of(evaluation.chemical_potentials.begin(), evaluation.chemical_potentials.end(),
                     [sentinel](double value) { return value == sentinel; }) &&
         std::all_of(evaluation.entropies.begin(), evaluation.entropies.end(),
                     [sentinel](double value) { return value == sentinel; }) &&
         std::all_of(evaluation.band_energies.begin(), evaluation.band_energies.end(),
                     [sentinel](double value) { return value == sentinel; }) &&
         std::all_of(evaluation.free_energies.begin(), evaluation.free_energies.end(),
                     [sentinel](double value) { return value == sentinel; });
}

bool factor(Evaluation& evaluation, const std::vector<double>& overlap, std::uint64_t generation,
            std::string& error) {
  return xtbloom::detail::gfn2::factor_overlap_cpu(evaluation.plan, overlap.data(), generation,
                                                   backend(), evaluation.scratch, evaluation.cache,
                                                   error) == XTBLOOM_STATUS_SUCCESS;
}

bool solve(Evaluation& evaluation, const std::vector<double>& hamiltonian, double temperature,
           std::uint64_t generation, std::string& error,
           const CpuLinearAlgebraBackend& selected_backend = backend()) {
  EigensolverThermodynamicsView thermodynamics = evaluation.thermodynamics();
  return xtbloom::detail::gfn2::solve_eigensystems_cpu(
             evaluation.plan, evaluation.cache, generation, hamiltonian.data(), temperature,
             selected_backend, evaluation.scratch, evaluation.wavefunction, thermodynamics,
             error) == XTBLOOM_STATUS_SUCCESS;
}

bool system_view(Evaluation& evaluation, std::size_t system, WavefunctionSystemView& view,
                 std::string& error) {
  return xtbloom::detail::gfn2::make_wavefunction_system_view(
             evaluation.layout, evaluation.wavefunction, static_cast<std::int64_t>(system), view,
             error) == XTBLOOM_STATUS_SUCCESS;
}

double metric_trace(const double* density, const double* overlap, std::size_t n) {
  double result = 0.0;
  for (std::size_t row = 0u; row < n; ++row) {
    for (std::size_t column = 0u; column < n; ++column) {
      result += density[row * n + column] * overlap[column * n + row];
    }
  }
  return result;
}

bool generalized_eigensystem_is_valid(const double* hamiltonian, const double* overlap,
                                      const double* coefficients, const double* eigenvalues,
                                      std::size_t n) {
  for (std::size_t orbital = 0u; orbital < n; ++orbital) {
    for (std::size_t row = 0u; row < n; ++row) {
      double hc = 0.0;
      double sc = 0.0;
      for (std::size_t column = 0u; column < n; ++column) {
        hc += hamiltonian[row * n + column] * coefficients[column * n + orbital];
        sc += overlap[row * n + column] * coefficients[column * n + orbital];
      }
      if (!near(hc, sc * eigenvalues[orbital], 8.0e-11)) {
        return false;
      }
    }
    for (std::size_t other = 0u; other < n; ++other) {
      double metric = 0.0;
      for (std::size_t row = 0u; row < n; ++row) {
        for (std::size_t column = 0u; column < n; ++column) {
          metric += coefficients[row * n + orbital] * overlap[row * n + column] *
                    coefficients[column * n + other];
        }
      }
      if (!near(metric, orbital == other ? 1.0 : 0.0, 8.0e-11)) {
        return false;
      }
    }
  }
  return true;
}

int test_occupations_degeneracy_and_fractional_filling() {
  std::string error;
  const std::array<double, 3> levels{{-1.0, 0.0, 1.0}};
  std::array<double, 3> occupations{{7.0, 7.0, 7.0}};
  double chemical_potential = 7.0;
  double entropy = 7.0;
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(3, levels.data(), 1.5, 0.0, occupations.data(),
                                                    chemical_potential, entropy,
                                                    error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(occupations[0] == 1.0 && occupations[1] == 0.5 && occupations[2] == 0.0);
  CHECK(chemical_potential == 0.0 && entropy == 0.0);

  const std::array<double, 2> degenerate{{0.0, 0.0}};
  std::array<double, 2> thermal{{0.0, 0.0}};
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(
            2, degenerate.data(), 1.0, XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE, thermal.data(),
            chemical_potential, entropy, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(near(thermal[0], 0.5, 2.0e-14) && near(thermal[1], 0.5, 2.0e-14));
  CHECK(near(thermal[0] + thermal[1], 1.0, 2.0e-14));
  CHECK(near(chemical_potential, 0.0, 2.0e-14));
  CHECK(near(entropy, 2.0 * std::log(2.0), 2.0e-13));

  const std::array<double, 3> triply_degenerate{{0.0, 0.0, 0.0}};
  std::array<double, 3> equal_fractional{{0.0, 0.0, 0.0}};
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(
            3, triply_degenerate.data(), 1.3, XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE,
            equal_fractional.data(), chemical_potential, entropy, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(equal_fractional[0] == equal_fractional[1] && equal_fractional[1] == equal_fractional[2]);
  CHECK(near(equal_fractional[0] + equal_fractional[1] + equal_fractional[2], 1.3, 2.0e-14));
  double entropy_from_published = 0.0;
  for (const double occupation : equal_fractional) {
    entropy_from_published -=
        occupation * std::log(occupation) + (1.0 - occupation) * std::log(1.0 - occupation);
  }
  CHECK(near(entropy, entropy_from_published, 2.0e-15));

  std::array<double, 3> fractional_thermal{{0.0, 0.0, 0.0}};
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(3, levels.data(), 1.3, 0.02,
                                                    fractional_thermal.data(), chemical_potential,
                                                    entropy, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(near(fractional_thermal[0] + fractional_thermal[1] + fractional_thermal[2], 1.3, 2.0e-13));
  CHECK(entropy > 0.0 && std::isfinite(chemical_potential));

  const std::array<double, 2> reversed{{1.0, -1.0}};
  const auto saved = thermal;
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(2, reversed.data(), 1.0, 0.0, thermal.data(),
                                                    chemical_potential, entropy,
                                                    error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(thermal == saved);

  const std::array<double, 2> extreme{
      {std::numeric_limits<double>::max() / 2.0, std::numeric_limits<double>::max()}};
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(
            2, extreme.data(), 1.0, std::numeric_limits<double>::max() / 256.0, thermal.data(),
            chemical_potential, entropy, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::isfinite(chemical_potential) && std::isfinite(entropy));
  CHECK(std::isfinite(thermal[0]) && std::isfinite(thermal[1]));
  CHECK(near(thermal[0] + thermal[1], 1.0, 8.0e-15));

  const std::array<double, 2> ordinary{{-3.0, 7.0}};
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(2, ordinary.data(), 0.0, 0.01, thermal.data(),
                                                    chemical_potential, entropy,
                                                    error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(thermal[0] == 0.0 && thermal[1] == 0.0);
  CHECK(chemical_potential == 0.0 && entropy == 0.0);
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(2, ordinary.data(), 2.0, 0.01, thermal.data(),
                                                    chemical_potential, entropy,
                                                    error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(thermal[0] == 1.0 && thermal[1] == 1.0);
  CHECK(std::isfinite(chemical_potential) && entropy == 0.0);

  std::array<double, 2> aliased_levels{{-1.0, 1.0}};
  const auto aliased_saved = aliased_levels;
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(
            2, aliased_levels.data(), 1.0, 0.01, aliased_levels.data(), chemical_potential, entropy,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(aliased_levels == aliased_saved);

  const std::array<double, 2> small_target_levels{{-0.2, 0.3}};
  for (const double target : {1.0e-16, 1.0e-20, 1.0e-300}) {
    CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(2, small_target_levels.data(), target, 0.01,
                                                      thermal.data(), chemical_potential, entropy,
                                                      error) == XTBLOOM_STATUS_SUCCESS);
    const double population = thermal[0] + thermal[1];
    CHECK(std::isfinite(population) && population > 0.0);
    CHECK(std::abs(population - target) <= 2.0e-13 * target);
    CHECK(std::isfinite(chemical_potential) && std::isfinite(entropy));
  }

  const std::array<double, 2> origin_levels{{0.0, 1.0}};
  const std::array<double, 2> translated_levels{{100.0, 101.0}};
  for (const double target : {1.0e-16, 1.0e-20, 1.0e-300}) {
    std::array<double, 2> origin_occupations{{0.0, 0.0}};
    std::array<double, 2> translated_occupations{{0.0, 0.0}};
    double origin_mu = 0.0;
    double origin_entropy = 0.0;
    CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(
              2, origin_levels.data(), target, 1.0e-7, origin_occupations.data(), origin_mu,
              origin_entropy, error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(
              2, translated_levels.data(), target, 1.0e-7, translated_occupations.data(),
              chemical_potential, entropy, error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(origin_occupations == translated_occupations);
    CHECK(origin_entropy == entropy);
    CHECK(near(chemical_potential, origin_mu + 100.0, 2.0e-13));
    const double translated_population = translated_occupations[0] + translated_occupations[1];
    CHECK(std::abs(translated_population - target) <= 2.0e-13 * target);
  }

  const double translated_near_full_target = std::nextafter(2.0, 0.0);
  std::array<double, 2> origin_near_full{{0.0, 0.0}};
  std::array<double, 2> translated_near_full{{0.0, 0.0}};
  double origin_near_full_mu = 0.0;
  double origin_near_full_entropy = 0.0;
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(
            2, origin_levels.data(), translated_near_full_target, 1.0e-7, origin_near_full.data(),
            origin_near_full_mu, origin_near_full_entropy, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(2, translated_levels.data(),
                                                    translated_near_full_target, 1.0e-7,
                                                    translated_near_full.data(), chemical_potential,
                                                    entropy, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(origin_near_full == translated_near_full);
  CHECK(origin_near_full_entropy == entropy);
  CHECK(near(chemical_potential, origin_near_full_mu + 100.0, 2.0e-13));
  const double translated_holes = (1.0 - translated_near_full[0]) + (1.0 - translated_near_full[1]);
  CHECK(std::abs(translated_holes - (2.0 - translated_near_full_target)) <=
        2.0e-13 * (2.0 - translated_near_full_target));

  const double near_full_target = std::nextafter(2.0, 0.0);
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(2, small_target_levels.data(), near_full_target,
                                                    0.01, thermal.data(), chemical_potential,
                                                    entropy, error) == XTBLOOM_STATUS_SUCCESS);
  const double holes = (1.0 - thermal[0]) + (1.0 - thermal[1]);
  const double expected_holes = 2.0 - near_full_target;
  CHECK(holes > 0.0 && std::abs(holes - expected_holes) <= 2.0e-13 * expected_holes);
  return 0;
}

int test_binary64_frontier_retry_controls() {
  namespace policy = xtbloom::detail::gfn2::binary64_policy;
  constexpr double temperature = 1.0e-7;

  policy::DoubleDouble just_outside{};
  policy::add_double_double(just_outside, std::nextafter(0.5, 0.0));
  policy::add_double_double(just_outside, -1.0);
  CHECK(!policy::double_double_within(just_outside, 0.5));
  policy::DoubleDouble exact_boundary{};
  policy::add_double_double(exact_boundary, -0.5);
  CHECK(policy::double_double_within(exact_boundary, 0.5));

  const std::array<double, 3> electron_levels{{0.0, 1.0, 1.0}};
  const double electron_target = std::nextafter(1.5, 0.0);
  policy::Root electron_root{};
  CHECK(policy::solve_root(electron_levels.data(), 3, electron_target, temperature, false,
                           electron_root));
  CHECK(electron_root.retried_at_frontier);
  CHECK(electron_root.energy_reference == 1.0);
  CHECK(electron_root.frontier_begin == 1 && electron_root.frontier_end == 3);
  policy::Publication electron_publication{};
  CHECK(policy::select_publication(electron_levels.data(), 3, electron_target, temperature, false,
                                   electron_root, electron_publication));
  CHECK(policy::audited_publication_within(electron_levels.data(), 3, temperature, electron_root,
                                           electron_publication, false, electron_target,
                                           64.0 * policy::kEpsilon * electron_target));
  CHECK(policy::published_occupation(electron_levels.data(), 1, temperature, electron_root,
                                     electron_publication) ==
        policy::published_occupation(electron_levels.data(), 2, temperature, electron_root,
                                     electron_publication));

  /* The same sharp frontier translated by a large common offset must retain
   * identical occupations and entropy; only the chemical potential translates. */
  const std::array<double, 3> translated_levels{{100.0, 101.0, 101.0}};
  policy::Root translated_root{};
  policy::Publication translated_publication{};
  CHECK(policy::solve_root(translated_levels.data(), 3, electron_target, temperature, false,
                           translated_root));
  CHECK(translated_root.retried_at_frontier);
  CHECK(policy::select_publication(translated_levels.data(), 3, electron_target, temperature, false,
                                   translated_root, translated_publication));
  for (std::int64_t orbital = 0; orbital < 3; ++orbital) {
    CHECK(policy::published_occupation(electron_levels.data(), orbital, temperature, electron_root,
                                       electron_publication) ==
          policy::published_occupation(translated_levels.data(), orbital, temperature,
                                       translated_root, translated_publication));
  }
  CHECK(policy::publication_entropy(electron_levels.data(), 3, temperature, electron_root,
                                    electron_publication) ==
        policy::publication_entropy(translated_levels.data(), 3, temperature, translated_root,
                                    translated_publication));
  const double electron_mu = policy::saturated_affine(electron_root.energy_reference,
                                                      electron_root.scaled_mu, temperature);
  const double translated_mu = policy::saturated_affine(translated_root.energy_reference,
                                                        translated_root.scaled_mu, temperature);
  CHECK(near(translated_mu, electron_mu + 100.0, 2.0e-15));

  /* Genuine hole mirror: the initial reference is the saturated higher
   * singleton, while the changing fractional frontier is the lower pair. */
  const std::array<double, 3> hole_levels{{0.0, 0.0, 1.0}};
  const double electron_count = std::nextafter(1.5, 3.0);
  const double hole_target = 3.0 - electron_count;
  policy::Root hole_root{};
  CHECK(policy::solve_root(hole_levels.data(), 3, hole_target, temperature, true, hole_root));
  CHECK(hole_root.retried_at_frontier);
  CHECK(hole_root.energy_reference == 0.0);
  CHECK(hole_root.frontier_begin == 0 && hole_root.frontier_end == 2);
  policy::Publication hole_publication{};
  CHECK(policy::select_publication(hole_levels.data(), 3, hole_target, temperature, true, hole_root,
                                   hole_publication));
  CHECK(policy::audited_publication_within(hole_levels.data(), 3, temperature, hole_root,
                                           hole_publication, true, hole_target,
                                           64.0 * policy::kEpsilon * hole_target));
  CHECK(policy::published_occupation(hole_levels.data(), 0, temperature, hole_root,
                                     hole_publication) ==
        policy::published_occupation(hole_levels.data(), 1, temperature, hole_root,
                                     hole_publication));

  policy::Root ordinary_root{};
  CHECK(policy::solve_root(electron_levels.data(), 3, 1.5, XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE,
                           false, ordinary_root));
  CHECK(!ordinary_root.retried_at_frontier);
  CHECK(policy::absolute(ordinary_root.quantity - 1.5) <= 64.0 * policy::kEpsilon * 1.5);

  policy::Root no_change{};
  no_change.energy_reference = 0.0;
  no_change.lower = 0.0;
  no_change.upper = std::nextafter(0.0, 1.0);
  CHECK(policy::unique_changing_degenerate_frontier(electron_levels.data(), 3, temperature, false,
                                                    no_change) == 3);

  const std::array<double, 4> ambiguous_levels{{0.0, 0.0, 1.0, 1.0}};
  policy::Root ambiguous{};
  ambiguous.energy_reference = 0.0;
  ambiguous.lower = -1.0e8;
  ambiguous.upper = 1.0e8;
  CHECK(policy::unique_changing_degenerate_frontier(ambiguous_levels.data(), 4, temperature, false,
                                                    ambiguous) == -1);

  const double denorm = std::numeric_limits<double>::denorm_min();
  const std::array<double, 2> reference_frontier{{1.0, 1.0}};
  policy::Root already_reference{};
  CHECK(policy::solve_root(reference_frontier.data(), 2, 9.0 * denorm,
                           XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE, false, already_reference));
  CHECK(!already_reference.retried_at_frontier);
  CHECK(already_reference.frontier_begin == 0 && already_reference.frontier_end == 2);
  CHECK(already_reference.energy_reference == 1.0);
  return 0;
}

/* Degenerate finite-temperature representability policy (#31).
 *
 * A symmetric (unitary- and permutation-invariant) description of an exactly
 * degenerate block requires every equal-energy orbital to share one binary64
 * occupation. When the requested electron/hole count falls between two such
 * symmetric states (three exactly degenerate orbitals with
 * nel = nextafter(3, 0)), the solver relaxes to the nearest representable
 * symmetric state instead of failing, with an absolute electron/hole error
 * bounded by the block quantization scale count * 2 * eps_double. */
int test_occupation_representability_policy() {
  std::string error;
  std::array<double, 3> occupations{{-1.0, -1.0, -1.0}};
  double chemical_potential = 7.0;
  double entropy = 7.0;
  const double eps = std::numeric_limits<double>::epsilon();

  /* Near-capacity limit: three exactly degenerate orbitals whose single-hole
   * count cannot be expressed as an equal triple of binary64 values. */
  const std::array<double, 3> near_capacity{{1.0, 1.0, 1.0}};
  const double near_capacity_nel = std::nextafter(3.0, 0.0);
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(
            3, near_capacity.data(), near_capacity_nel, XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE,
            occupations.data(), chemical_potential, entropy, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(occupations[0] == occupations[1] && occupations[1] == occupations[2]);
  CHECK(occupations[0] > 0.0 && occupations[0] <= 1.0);
  const double one_hole_state = std::nextafter(1.0, 0.0);
  CHECK(one_hole_state == 0x1.fffffffffffffp-1);
  CHECK(occupations[0] == one_hole_state);
  const long double near_capacity_sum = static_cast<long double>(occupations[0]) +
                                        static_cast<long double>(occupations[1]) +
                                        static_cast<long double>(occupations[2]);
  const long double near_capacity_error =
      std::abs(near_capacity_sum - static_cast<long double>(near_capacity_nel));
  CHECK(near_capacity_error <= 2.0L * static_cast<long double>(eps) * 3.0L);
  /* Entropy is derived from the published occupations, never from an idealized
   * occupation that differs from what fill_occupations_cpu publishes. */
  double published_entropy = 0.0;
  for (const double occupation : occupations) {
    if (occupation > 0.0 && occupation < 1.0) {
      published_entropy -=
          occupation * std::log(occupation) + (1.0 - occupation) * std::log(1.0 - occupation);
    }
  }
  CHECK(std::isfinite(entropy) && std::abs(entropy - published_entropy) <= 1.0e-15);

  /* Exact half-subnormal tie: 9 * denorm_min split symmetrically over two
   * orbitals lies halfway between 4 and 5 denorm_min per member. The canonical
   * backend-independent rule chooses the lower occupation. */
  const double denorm = std::numeric_limits<double>::denorm_min();
  const std::array<double, 2> tie_levels{{1.0, 1.0}};
  std::array<double, 2> tie_occupations{{-1.0, -1.0}};
  const double tie_target = 9.0 * denorm;
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(
            2, tie_levels.data(), tie_target, XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE,
            tie_occupations.data(), chemical_potential, entropy, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(tie_occupations[0] == 4.0 * denorm && tie_occupations[1] == 4.0 * denorm);
  const long double lower_error =
      std::abs(8.0L * static_cast<long double>(denorm) - static_cast<long double>(tie_target));
  const long double upper_error =
      std::abs(10.0L * static_cast<long double>(denorm) - static_cast<long double>(tie_target));
  CHECK(lower_error == upper_error);
  const long double tie_occupation = static_cast<long double>(tie_occupations[0]);
  const double tie_entropy =
      static_cast<double>(-2.0L * (tie_occupation * std::log(tie_occupation) +
                                   (1.0L - tie_occupation) * std::log(1.0L - tie_occupation)));
  CHECK(entropy == tie_entropy);

  /* Electron-poor limit: a subnormal electron target on the same degenerate
   * block is also relaxed to the nearest symmetric (all-zero) state. */
  std::array<double, 3> poor{{-1.0, -1.0, -1.0}};
  const double electron_poor_nel = std::nextafter(0.0, 1.0);
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(
            3, near_capacity.data(), electron_poor_nel, XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE,
            poor.data(), chemical_potential, entropy, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(poor[0] == poor[1] && poor[1] == poor[2]);
  const double electron_poor_error = std::abs((poor[0] + poor[1] + poor[2]) - electron_poor_nel);
  CHECK(electron_poor_error <= 2.0 * eps * 3.0);
  CHECK(entropy == 0.0 && std::isfinite(chemical_potential));

  /* Root spacing for this subnormal target is exhausted at the lower
   * two-member frontier, but publication can conserve strictly by placing the
   * denormal electron on the higher singleton. The causal frontier floor must
   * survive that later strict rescue. */
  const std::array<double, 3> causal_frontier_rescue{{0.0, 0.0, 1.0}};
  std::array<double, 3> causal_frontier_occupations{{-1.0, -1.0, -1.0}};
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(
            3, causal_frontier_rescue.data(), electron_poor_nel, 1.0e-7,
            causal_frontier_occupations.data(), chemical_potential, entropy,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(causal_frontier_occupations[0] == 0.0 && causal_frontier_occupations[1] == 0.0 &&
        causal_frontier_occupations[2] == electron_poor_nel);
  CHECK(entropy > 0.0 && std::isfinite(entropy) && std::isfinite(chemical_potential));

  /* The canonical subnormal logit retains the existing saturated chemical-
   * potential contract even when the translated offset itself exceeds the
   * binary64 range. */
  const double maximum = std::numeric_limits<double>::max();
  const std::array<double, 3> extreme_translated{{maximum, maximum, maximum}};
  std::array<double, 3> extreme_occupations{{-1.0, -1.0, -1.0}};
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(
            3, extreme_translated.data(), electron_poor_nel, maximum, extreme_occupations.data(),
            chemical_potential, entropy, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(extreme_occupations[0] == 0.0 && extreme_occupations[1] == 0.0 &&
        extreme_occupations[2] == 0.0);
  CHECK(chemical_potential == -maximum);
  CHECK(entropy == 0.0);

  /* An exactly representable symmetric target keeps its exact arithmetic
   * (this path pre-dates the relaxation and must stay byte-identical). */
  const std::array<double, 2> exact_pair{{1.0, 1.0}};
  std::array<double, 2> exact_occupations{{-1.0, -1.0}};
  const double exact_nel = std::nextafter(2.0, 0.0);
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(2, exact_pair.data(), exact_nel,
                                                    XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE,
                                                    exact_occupations.data(), chemical_potential,
                                                    entropy, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(exact_occupations[0] == exact_occupations[1]);
  CHECK(std::abs((exact_occupations[0] + exact_occupations[1]) - exact_nel) == 0.0);

  /* A singleton below a three-member frontier block can absorb the remaining
   * one-hole ulp exactly. Strict global candidates must take precedence over
   * the relaxed three-member state. */
  const std::array<double, 4> partial_strict{{0.0, 1.0, 1.0, 1.0}};
  std::array<double, 4> partial_strict_occupations{};
  const double partial_strict_nel = std::nextafter(4.0, 0.0);
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(
            4, partial_strict.data(), partial_strict_nel, XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE,
            partial_strict_occupations.data(), chemical_potential, entropy,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::all_of(partial_strict_occupations.begin(), partial_strict_occupations.end(),
                    [one_hole_state](double occupation) { return occupation == one_hole_state; }));
  long double partial_strict_holes = 0.0L;
  for (const double occupation : partial_strict_occupations) {
    partial_strict_holes += 1.0L - static_cast<long double>(occupation);
  }
  CHECK(partial_strict_holes == 4.0L - static_cast<long double>(partial_strict_nel));

  /* A sharp low-temperature degenerate frontier can exhaust root spacing
   * when the translated frame is anchored at an extreme level. Retrying at
   * the actual frontier must recover the ordinary strict root. */
  const std::array<double, 3> low_temperature_frontier{{0.0, 1.0, 1.0}};
  std::array<double, 3> low_temperature_occupations{};
  const double low_temperature_nel = std::nextafter(1.5, 0.0);
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(
            3, low_temperature_frontier.data(), low_temperature_nel, 1.0e-7,
            low_temperature_occupations.data(), chemical_potential, entropy,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(low_temperature_occupations[1] == low_temperature_occupations[2]);
  long double low_temperature_sum = 0.0L;
  for (const double occupation : low_temperature_occupations) {
    low_temperature_sum += static_cast<long double>(occupation);
  }
  CHECK(std::abs(low_temperature_sum - static_cast<long double>(low_temperature_nel)) <=
        64.0L * static_cast<long double>(eps) * static_cast<long double>(low_temperature_nel));

  const std::array<double, 6> broad_low_temperature_frontier{{0.0, 1.0, 1.0, 1.0, 1.0, 1.0}};
  std::array<double, 6> broad_low_temperature_occupations{};
  const double broad_low_temperature_nel = std::nextafter(3.0, 0.0);
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(
            6, broad_low_temperature_frontier.data(), broad_low_temperature_nel, 1.0e-7,
            broad_low_temperature_occupations.data(), chemical_potential, entropy,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::all_of(broad_low_temperature_occupations.begin() + 1,
                    broad_low_temperature_occupations.end(),
                    [frontier = broad_low_temperature_occupations[1]](double occupation) {
                      return occupation == frontier;
                    }));

  /* Without an actual multi-orbital degeneracy block, the relaxed tolerance
   * is unavailable. The ordinary singleton correction path must conserve the
   * same near-capacity target at the strict publication tolerance. */
  const std::array<double, 3> nondegenerate{{0.0, 1.0, 2.0}};
  std::array<double, 3> nondegenerate_occupations{{-1.0, -1.0, -1.0}};
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(
            3, nondegenerate.data(), near_capacity_nel, XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE,
            nondegenerate_occupations.data(), chemical_potential, entropy,
            error) == XTBLOOM_STATUS_SUCCESS);
  const double nondegenerate_holes = (1.0 - nondegenerate_occupations[0]) +
                                     (1.0 - nondegenerate_occupations[1]) +
                                     (1.0 - nondegenerate_occupations[2]);
  const double target_holes = 3.0 - near_capacity_nel;
  CHECK(std::abs(nondegenerate_holes - target_holes) <= 64.0 * eps * target_holes);

  /* Two exact-degeneracy blocks exercise the relaxed mixed-spectrum path.
   * Three low levels remain full while three equal frontier levels publish the
   * closest common binary64 value. The target hole count is eight binary64
   * ulps below capacity; the frontier block can publish six or nine such ulps,
   * so nine is the unique nearest symmetric state. */
  const std::array<double, 6> mixed_degenerate{{0.0, 0.0, 0.0, 1.0, 1.0, 1.0}};
  std::array<double, 6> mixed_degenerate_occupations{};
  const double mixed_degenerate_nel = std::nextafter(6.0, 0.0);
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(
            6, mixed_degenerate.data(), mixed_degenerate_nel,
            XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE, mixed_degenerate_occupations.data(),
            chemical_potential, entropy, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::all_of(mixed_degenerate_occupations.begin(), mixed_degenerate_occupations.begin() + 3,
                    [](double occupation) { return occupation == 1.0; }));
  const double three_hole_state =
      std::nextafter(std::nextafter(std::nextafter(1.0, 0.0), 0.0), 0.0);
  CHECK(three_hole_state == 0x1.ffffffffffffdp-1);
  CHECK(std::all_of(
      mixed_degenerate_occupations.begin() + 3, mixed_degenerate_occupations.end(),
      [three_hole_state](double occupation) { return occupation == three_hole_state; }));
  long double mixed_degenerate_sum = 0.0L;
  double mixed_degenerate_entropy = 0.0;
  for (const double occupation : mixed_degenerate_occupations) {
    mixed_degenerate_sum += static_cast<long double>(occupation);
    if (occupation > 0.0 && occupation < 1.0) {
      mixed_degenerate_entropy -=
          occupation * std::log(occupation) + (1.0 - occupation) * std::log(1.0 - occupation);
    }
  }
  const long double mixed_degenerate_error =
      std::abs(mixed_degenerate_sum - static_cast<long double>(mixed_degenerate_nel));
  CHECK(mixed_degenerate_error <= 2.0L * static_cast<long double>(eps) * 3.0L);
  CHECK(std::abs(entropy - mixed_degenerate_entropy) <= 1.0e-15);
  const double two_hole_state = std::nextafter(std::nextafter(1.0, 0.0), 0.0);
  const long double adjacent_error =
      std::abs(3.0L + 3.0L * static_cast<long double>(two_hole_state) -
               static_cast<long double>(mixed_degenerate_nel));
  CHECK(mixed_degenerate_error < adjacent_error);

  /* Degenerate members embedded among non-degenerate levels keep equal
   * occupations, so relabelling the block cannot change published values. */
  const std::array<double, 4> mixed{{0.0, 1.0, 1.0, 2.0}};
  std::array<double, 4> mixed_occupations{{-1.0, -1.0, -1.0, -1.0}};
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(
            4, mixed.data(), 1.3, XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE, mixed_occupations.data(),
            chemical_potential, entropy, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(mixed_occupations[1] == mixed_occupations[2]);

  /* The relaxation must not fabricate success for an impossible input: inputs
   * that are structurally invalid remain deterministic invalid arguments. */
  const std::array<double, 3> reversed{{1.0, 0.0, -1.0}};
  const auto saved = occupations;
  const double saved_chemical_potential = chemical_potential;
  const double saved_entropy = entropy;
  CHECK(xtbloom::detail::gfn2::fill_occupations_cpu(3, reversed.data(), 1.5, 0.0,
                                                    occupations.data(), chemical_potential, entropy,
                                                    error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(occupations == saved);
  CHECK(chemical_potential == saved_chemical_potential);
  CHECK(entropy == saved_entropy);
  return 0;
}

int test_production_lp64_factory() {
  CpuLinearAlgebraBackend production;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_mkl_rt_lp64_backend(production, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(production.ready());
  /* Any verified lazily-loaded LP64 backend (MKL or OpenBLAS) qualifies as
   * production; production_mkl() additionally distinguishes the MKL provider. */
  CHECK(production.production());
  CHECK(error.empty());
  return 0;
}

int run_mkl_ilp64_rejection_child() {
  CpuLinearAlgebraBackend production;
  std::string error;
  const xtbloom_status_t status =
      xtbloom::detail::gfn2::make_mkl_rt_lp64_backend(production, error);
  if (status == XTBLOOM_STATUS_SUCCESS && production.ready() && !production.production_mkl()) {
    /* An OpenBLAS runtime has no MKL interface layer to force to ILP64, so
     * there is nothing to reject; accept the vacuous success. */
    return 0;
  }
  if (status == XTBLOOM_STATUS_SUCCESS && production.ready() &&
      production.production_mkl_isolated()) {
    /* The host-isolated MKL shim never reads MKL_INTERFACE_LAYER and never
     * switches the embedding process's MKL interface layer, so an explicit
     * ILP64 environment is ignored and LP64 correctness is preserved. This is
     * the required coexistence outcome for issue #30, not a rejection. */
    return 0;
  }
  return status == XTBLOOM_STATUS_BACKEND_UNAVAILABLE && !production.ready() &&
                 error.find("ILP64") != std::string::npos
             ? 0
             : 1;
}

int test_mkl_ilp64_rejection_in_fresh_process() {
  char executable_path[4096]{};
  const ssize_t executable_path_size =
      readlink("/proc/self/exe", executable_path, sizeof(executable_path) - 1u);
  CHECK(executable_path_size > 0);
  executable_path[static_cast<std::size_t>(executable_path_size)] = '\0';

  const pid_t child = fork();
  CHECK(child >= 0);
  if (child == 0) {
    if (setenv("MKL_INTERFACE_LAYER", "ILP64", 1) != 0) {
      _exit(120);
    }
    /* The isolated provider is resolved beside the executable in this static
     * unit-test binary, so preserve that real sibling directory across exec. */
    execl(executable_path, executable_path, "--mkl-ilp64-rejection-child",
          static_cast<char*>(nullptr));
    _exit(121);
  }

  int child_status = 0;
  pid_t waited = -1;
  do {
    waited = waitpid(child, &child_status, 0);
  } while (waited < 0 && errno == EINTR);
  CHECK(waited == child);
  CHECK(WIFEXITED(child_status));
  CHECK(WEXITSTATUS(child_status) == 0);
  return 0;
}

int test_unrestricted_literal_generalized_eigenproblem() {
  std::string error;
  Evaluation evaluation;
  CHECK(initialize_evaluation({0, 2}, {1, 1}, {0.0}, {0}, {2}, evaluation, error));
  const std::vector<double> overlap{1.2, 0.15, 0.15, 0.9};
  const std::vector<double> hamiltonian{-0.8, 0.13, 0.13, 0.25, -0.55, -0.08, -0.08, 0.42};
  CHECK(factor(evaluation, overlap, 73u, error));
  fill_outputs(evaluation, 91.0);
  CHECK(solve(evaluation, hamiltonian, 0.0, 73u, error));
  WavefunctionSystemView view;
  CHECK(system_view(evaluation, 0u, view, error));

  /* Literals are independent generalized-symmetric 2x2 reference values. */
  CHECK(near(view.eigenvalues[0], -0.7192210550444913, 3.0e-13));
  CHECK(near(view.eigenvalues[1], 0.28517850185300192, 3.0e-13));
  CHECK(near(view.eigenvalues[2], -0.45845957914509017, 3.0e-13));
  CHECK(near(view.eigenvalues[3], 0.48966525290395535, 3.0e-13));
  CHECK(generalized_eigensystem_is_valid(hamiltonian.data(), overlap.data(), view.coefficients,
                                         view.eigenvalues, 2u));
  CHECK(generalized_eigensystem_is_valid(hamiltonian.data() + 4u, overlap.data(),
                                         view.coefficients + 4u, view.eigenvalues + 2u, 2u));
  CHECK(near(evaluation.band_energies[0], -1.1776806341895814, 4.0e-13));
  CHECK(near(metric_trace(view.density, overlap.data(), 2u), 1.0, 3.0e-13));
  CHECK(near(metric_trace(view.density + 4u, overlap.data(), 2u), 1.0, 3.0e-13));
  return 0;
}

int test_tblite_300_kelvin_pinned_oracle() {
  std::string error;
  Evaluation evaluation;
  CHECK(initialize_evaluation({0, 3}, {1, 1, 1}, {0.6}, {0}, {1}, evaluation, error));
  const std::vector<double> overlap{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const std::vector<double> hamiltonian{-0.0012, 0.0, 0.0, 0.0, 0.0007, 0.0, 0.0, 0.0, 0.0029};
  CHECK(factor(evaluation, overlap, 79u, error));
  fill_outputs(evaluation, 93.0);
  CHECK(solve(evaluation, hamiltonian, XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE, 79u, error));
  WavefunctionSystemView view;
  CHECK(system_view(evaluation, 0u, view, error));

  /* Pinned from tblite get_fermi_filling for nel=1.2 and this asymmetric spectrum. */
  constexpr std::array<double, 3> expected_occupations{
      {0.79928021315435582, 0.35021173693263352, 0.050508049913011548}};
  for (std::size_t orbital = 0u; orbital < expected_occupations.size(); ++orbital) {
    CHECK(near(view.occupations[orbital], expected_occupations[orbital], 8.0e-13));
    CHECK(near(view.occupations[3u + orbital], expected_occupations[orbital], 8.0e-13));
  }
  CHECK(near(evaluation.chemical_potentials[0], 0.00011277048977345888, 8.0e-13));
  CHECK(near(evaluation.chemical_potentials[1], 0.00011277048977345888, 8.0e-13));
  CHECK(near(evaluation.entropies[0], 2.6979694267209249, 2.0e-12));
  CHECK(near(evaluation.band_energies[0], -0.0011350293903693001, 2.0e-14));
  CHECK(near(evaluation.free_energies[0], -0.0036982152079269824, 2.0e-14));
  return 0;
}

int test_restricted_closed_shell_and_warm_reuse() {
  std::string error;
#if !defined(XTBLOOM_EIGENSOLVER_TEST_ASAN) && defined(__GLIBC__)
  allocation_test::malloc_count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  void* const allocation_probe = std::malloc(17u);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  CHECK(allocation_probe != nullptr);
  std::free(allocation_probe);
  CHECK(allocation_test::malloc_count.load(std::memory_order_relaxed) == 1u);
#endif
  Evaluation evaluation;
  CHECK(initialize_evaluation({0, 2}, {1, 1}, {0.0}, {0}, {1}, evaluation, error));
  const std::vector<double> overlap{1.0, 0.2, 0.2, 1.0};
  const std::vector<double> hamiltonian{-0.8, 0.1, 0.1, 0.3};
  reset_backend_spies();
  CHECK(factor(evaluation, overlap, 41u, error));
  CHECK(potrf_calls.load(std::memory_order_relaxed) == 1);
  CHECK(dpocon_calls.load(std::memory_order_relaxed) == 1);
  fill_outputs(evaluation, 77.0);
  CHECK(solve(evaluation, hamiltonian, 0.0, 41u, error));
  CHECK(evaluation.statuses[0] == XTBLOOM_STATUS_SUCCESS);
  WavefunctionSystemView view;
  CHECK(system_view(evaluation, 0u, view, error));
  CHECK(generalized_eigensystem_is_valid(hamiltonian.data(), overlap.data(), view.coefficients,
                                         view.eigenvalues, 2u));
  CHECK(near(view.occupations[0] + view.occupations[1], 1.0));
  CHECK(near(view.occupations[2] + view.occupations[3], 1.0));
  CHECK(near(metric_trace(view.density, overlap.data(), 2u), 2.0));
  CHECK(near(evaluation.band_energies[0],
             view.eigenvalues[0] * (view.occupations[0] + view.occupations[2]) +
                 view.eigenvalues[1] * (view.occupations[1] + view.occupations[3])));
  CHECK(near(metric_trace(view.energy_weighted_density, overlap.data(), 2u),
             evaluation.band_energies[0]));
  CHECK(near(metric_trace(view.density, hamiltonian.data(), 2u), evaluation.band_energies[0]));
  CHECK(evaluation.entropies[0] == 0.0);
  CHECK(evaluation.free_energies[0] == evaluation.band_energies[0]);
  CHECK(dsyevd_calls.load(std::memory_order_relaxed) == 1);
  CHECK(dtrsm_calls.load(std::memory_order_relaxed) == 3);
  CHECK(dgemm_calls.load(std::memory_order_relaxed) == 2);
  CHECK(non_column_major_calls.load(std::memory_order_relaxed) == 0);

  /* Warm both MKL paths before observing steady-state allocations. */
  CHECK(factor(evaluation, overlap, 41u, error));
  CHECK(solve(evaluation, hamiltonian, 0.0, 41u, error));
  allocation_test::new_count.store(0u, std::memory_order_relaxed);
  allocation_test::malloc_count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const bool second_factor = factor(evaluation, overlap, 41u, error);
  const bool second_solve = solve(evaluation, hamiltonian, 0.0, 41u, error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  CHECK(second_factor);
  CHECK(second_solve);
  CHECK(allocation_test::new_count.load(std::memory_order_relaxed) == 0u);
#if !defined(XTBLOOM_EIGENSOLVER_TEST_ASAN) && defined(__GLIBC__)
  CHECK(allocation_test::malloc_count.load(std::memory_order_relaxed) == 0u);
#endif
  CHECK(non_column_major_calls.load(std::memory_order_relaxed) == 0);
  return 0;
}

int test_open_shell_restricted_radical_and_spin_singlet() {
  std::string error;
  const std::vector<double> one_overlap{1.0};
  const std::vector<double> one_hamiltonian{-0.5};

  Evaluation restricted;
  CHECK(initialize_evaluation({0, 1}, {1}, {0.0}, {1}, {1}, restricted, error));
  CHECK(factor(restricted, one_overlap, 5u, error));
  fill_outputs(restricted, 17.0);
  CHECK(solve(restricted, one_hamiltonian, 0.0, 5u, error));
  WavefunctionSystemView restricted_view;
  CHECK(system_view(restricted, 0u, restricted_view, error));
  CHECK(restricted_view.occupations[0] == 1.0 && restricted_view.occupations[1] == 0.0);
  CHECK(near(restricted_view.density[0], 1.0));

  Evaluation unrestricted;
  CHECK(initialize_evaluation({0, 1}, {1}, {0.0}, {1}, {2}, unrestricted, error));
  CHECK(factor(unrestricted, one_overlap, 6u, error));
  fill_outputs(unrestricted, 17.0);
  const std::vector<double> two_spin_hamiltonian{-0.5, -0.4};
  CHECK(solve(unrestricted, two_spin_hamiltonian, 0.0, 6u, error));
  WavefunctionSystemView unrestricted_view;
  CHECK(system_view(unrestricted, 0u, unrestricted_view, error));
  CHECK(unrestricted_view.occupations[0] == 1.0 && unrestricted_view.occupations[1] == 0.0);
  CHECK(near(unrestricted_view.density[0], 1.0));
  CHECK(near(unrestricted_view.density[1], 0.0));

  Evaluation singlet;
  CHECK(initialize_evaluation({0, 2}, {1, 1}, {0.0}, {0}, {2}, singlet, error));
  const std::vector<double> two_overlap{1.0, 0.0, 0.0, 1.0};
  const std::vector<double> singlet_hamiltonian{-0.7, 0.1, 0.1, 0.2, -0.7, 0.1, 0.1, 0.2};
  CHECK(factor(singlet, two_overlap, 7u, error));
  fill_outputs(singlet, 17.0);
  CHECK(solve(singlet, singlet_hamiltonian, 0.0, 7u, error));
  WavefunctionSystemView singlet_view;
  CHECK(system_view(singlet, 0u, singlet_view, error));
  CHECK(near(singlet_view.occupations[0] + singlet_view.occupations[1], 1.0));
  CHECK(near(singlet_view.occupations[2] + singlet_view.occupations[3], 1.0));
  for (std::size_t element = 0u; element < 4u; ++element) {
    CHECK(near(singlet_view.density[element], singlet_view.density[4u + element]));
  }

  Evaluation fractional;
  CHECK(initialize_evaluation({0, 2}, {1, 1}, {0.5}, {0}, {1}, fractional, error));
  CHECK(factor(fractional, two_overlap, 8u, error));
  fill_outputs(fractional, 17.0);
  CHECK(solve(fractional, {-0.7, 0.0, 0.0, 0.2}, 0.0, 8u, error));
  WavefunctionSystemView fractional_view;
  CHECK(system_view(fractional, 0u, fractional_view, error));
  CHECK(near(fractional_view.occupations[0] + fractional_view.occupations[1], 0.75));
  CHECK(near(fractional_view.occupations[2] + fractional_view.occupations[3], 0.75));
  CHECK(near(metric_trace(fractional_view.density, two_overlap.data(), 2u), 1.5));
  return 0;
}

int test_finite_temperature_free_energy() {
  std::string error;
  Evaluation evaluation;
  CHECK(initialize_evaluation({0, 2}, {1, 1}, {0.0}, {0}, {1}, evaluation, error));
  const std::vector<double> overlap{1.0, 0.0, 0.0, 1.0};
  const std::vector<double> hamiltonian{-0.001, 0.0, 0.0, 0.001};
  CHECK(factor(evaluation, overlap, 11u, error));
  fill_outputs(evaluation, 23.0);
  CHECK(solve(evaluation, hamiltonian, XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE, 11u, error));
  WavefunctionSystemView view;
  CHECK(system_view(evaluation, 0u, view, error));
  CHECK(near(view.occupations[0] + view.occupations[1], 1.0, 4.0e-14));
  CHECK(near(view.occupations[2] + view.occupations[3], 1.0, 4.0e-14));
  CHECK(near(evaluation.chemical_potentials[0], 0.0, 4.0e-14));
  CHECK(near(evaluation.chemical_potentials[1], 0.0, 4.0e-14));
  CHECK(evaluation.entropies[0] > 0.0);
  CHECK(near(evaluation.free_energies[0],
             evaluation.band_energies[0] -
                 XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE * evaluation.entropies[0],
             2.0e-14));
  CHECK(near(metric_trace(view.density, overlap.data(), 2u), 2.0, 8.0e-14));
  CHECK(near(metric_trace(view.energy_weighted_density, overlap.data(), 2u),
             evaluation.band_energies[0], 8.0e-14));
  return 0;
}

int test_ragged_failure_and_atomicity() {
  std::string error;
  Evaluation evaluation;
  CHECK(initialize_evaluation({0, 2, 4}, {1, 1, 1, 1}, {0.0, 0.0}, {0, 0}, {1, 1}, evaluation,
                              error, 1.0e-10));
  const double almost_one = 1.0 - 1.0e-14;
  const std::vector<double> overlap{1.0, 0.0, 0.0, 1.0, 1.0, almost_one, almost_one, 1.0};
  std::fill_n(evaluation.cache.cholesky_factors,
              static_cast<std::size_t>(evaluation.plan.total_matrix_elements()), 31.0);
  std::fill_n(evaluation.cache.geometry_generations,
              static_cast<std::size_t>(evaluation.plan.batch_size()), 101u);
  std::fill_n(evaluation.cache.system_statuses,
              static_cast<std::size_t>(evaluation.plan.batch_size()),
              static_cast<xtbloom_status_t>(77));
  std::vector<double> invalid_overlap = overlap;
  invalid_overlap.back() = std::numeric_limits<double>::quiet_NaN();
  CHECK(xtbloom::detail::gfn2::factor_overlap_cpu(evaluation.plan, invalid_overlap.data(), 17u,
                                                  backend(), evaluation.scratch, evaluation.cache,
                                                  error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(evaluation.cache.cholesky_factors,
                    evaluation.cache.cholesky_factors + evaluation.plan.total_matrix_elements(),
                    [](double value) { return value == 31.0; }));
  CHECK(evaluation.cache.geometry_generations[0] == 101u &&
        evaluation.cache.geometry_generations[1] == 101u);
  CHECK(evaluation.cache.system_statuses[0] == 77 && evaluation.cache.system_statuses[1] == 77);
  CHECK(factor(evaluation, overlap, 17u, error));
  CHECK(evaluation.cache.system_statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(evaluation.cache.system_statuses[1] == XTBLOOM_STATUS_EIGENSOLVER_FAILED);
  for (std::size_t element = 4u; element < 8u; ++element) {
    CHECK(evaluation.cache.cholesky_factors[element] == 31.0);
  }

  const std::vector<double> hamiltonian{-0.5, 0.0, 0.0, 0.2, -0.4, 0.0, 0.0, 0.3};
  fill_outputs(evaluation, 43.0);
  CHECK(solve(evaluation, hamiltonian, 0.0, 17u, error));
  CHECK(evaluation.statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(evaluation.statuses[1] == XTBLOOM_STATUS_EIGENSOLVER_FAILED);
  WavefunctionSystemView failed;
  CHECK(system_view(evaluation, 1u, failed, error));
  CHECK(std::all_of(failed.coefficients, failed.coefficients + 4,
                    [](double value) { return value == 43.0; }));
  CHECK(
      std::all_of(failed.density, failed.density + 4, [](double value) { return value == 43.0; }));
  CHECK(evaluation.entropies[1] == 43.0 && evaluation.band_energies[1] == 43.0 &&
        evaluation.free_energies[1] == 43.0);

  fill_outputs(evaluation, 53.0);
  EigensolverThermodynamicsView alias = evaluation.thermodynamics();
  alias.band_energies = evaluation.wavefunction.coefficients;
  const std::vector<double> saved_coefficients(
      evaluation.wavefunction.coefficients,
      evaluation.wavefunction.coefficients + evaluation.layout.coefficients.element_count);
  CHECK(xtbloom::detail::gfn2::solve_eigensystems_cpu(
            evaluation.plan, evaluation.cache, 17u, hamiltonian.data(), 0.0, backend(),
            evaluation.scratch, evaluation.wavefunction, alias,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::equal(saved_coefficients.begin(), saved_coefficients.end(),
                   evaluation.wavefunction.coefficients));
  CHECK(std::all_of(evaluation.statuses.begin(), evaluation.statuses.end(),
                    [](xtbloom_status_t value) { return value == 99; }));

  AlignedBuffer oversized(evaluation.plan.workspace_size_bytes() +
                          xtbloom::detail::gfn2::kEigensolverWorkspaceAlignment);
  EigensolverWorkspace rejected = evaluation.scratch;
  CHECK(xtbloom::detail::gfn2::bind_eigensolver_workspace(
            evaluation.plan, static_cast<std::byte*>(oversized.data) + 1u,
            evaluation.plan.workspace_size_bytes(), rejected,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(rejected.workspace_base == evaluation.scratch.workspace_base &&
        rejected.coefficients == evaluation.scratch.coefficients &&
        rejected.lapack_work == evaluation.scratch.lapack_work);
  return 0;
}

int test_plan_identity_generation_and_alias_rejection() {
  std::string error;
  Evaluation evaluation;
  CHECK(initialize_evaluation({0, 2}, {1, 1}, {0.0}, {0}, {1}, evaluation, error));
  const std::vector<double> overlap{1.0, 0.2, 0.2, 1.0};
  const std::vector<double> hamiltonian{-0.8, 0.1, 0.1, 0.3};
  CHECK(factor(evaluation, overlap, 83u, error));

  static_assert(std::is_nothrow_copy_constructible_v<EigensolverPlan>);
  EigensolverPlan copied = evaluation.plan;
  CHECK(copied.identity() == evaluation.plan.identity());
  fill_outputs(evaluation, 101.0);
  EigensolverThermodynamicsView thermodynamics = evaluation.thermodynamics();
  CHECK(xtbloom::detail::gfn2::solve_eigensystems_cpu(
            copied, evaluation.cache, 83u, hamiltonian.data(), 0.0, backend(), evaluation.scratch,
            evaluation.wavefunction, thermodynamics, error) == XTBLOOM_STATUS_SUCCESS);

  fill_outputs(evaluation, 103.0);
  CHECK(xtbloom::detail::gfn2::solve_eigensystems_cpu(
            copied, evaluation.cache, 84u, hamiltonian.data(), 0.0, backend(), evaluation.scratch,
            evaluation.wavefunction, thermodynamics, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(evaluation.statuses[0] == XTBLOOM_STATUS_EIGENSOLVER_FAILED);
  CHECK(evaluation.band_energies[0] == 103.0);

  Evaluation cross;
  CHECK(initialize_evaluation({0, 2}, {1, 1}, {0.0}, {0}, {1}, cross, error));
  CHECK(cross.plan.identity() != evaluation.plan.identity());
  CHECK(xtbloom::detail::gfn2::factor_overlap_cpu(cross.plan, overlap.data(), 83u, backend(),
                                                  evaluation.scratch, evaluation.cache,
                                                  error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  EigensolverPlan source = evaluation.plan;
  EigensolverPlan moved = std::move(source);
  CHECK(!source.sealed());
  CHECK(moved.identity() == evaluation.plan.identity());
  CHECK(xtbloom::detail::gfn2::factor_overlap_cpu(source, overlap.data(), 83u, backend(),
                                                  evaluation.scratch, evaluation.cache,
                                                  error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  EigensolverPlan empty;
  CHECK(xtbloom::detail::gfn2::factor_overlap_cpu(empty, overlap.data(), 83u, backend(),
                                                  evaluation.scratch, evaluation.cache,
                                                  error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  CHECK(xtbloom::detail::gfn2::factor_overlap_cpu(
            evaluation.plan, evaluation.cache.cholesky_factors + 1u, 83u, backend(),
            evaluation.scratch, evaluation.cache, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::factor_overlap_cpu(
            evaluation.plan, reinterpret_cast<const double*>(&evaluation.cache), 83u, backend(),
            evaluation.scratch, evaluation.cache, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  AlignedBuffer shared_storage(
      std::max(evaluation.plan.overlap_cache_size_bytes(), evaluation.plan.workspace_size_bytes()));
  EigensolverOverlapCache shared_cache;
  EigensolverWorkspace shared_scratch;
  CHECK(xtbloom::detail::gfn2::bind_eigensolver_overlap_cache(evaluation.plan, shared_storage.data,
                                                              shared_storage.size, shared_cache,
                                                              error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::bind_eigensolver_workspace(evaluation.plan, shared_storage.data,
                                                          shared_storage.size, shared_scratch,
                                                          error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::factor_overlap_cpu(evaluation.plan, overlap.data(), 83u, backend(),
                                                  shared_scratch, shared_cache,
                                                  error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  thermodynamics = evaluation.thermodynamics();
  thermodynamics.entropies = const_cast<double*>(evaluation.plan.alpha_electron_counts().data());
  CHECK(xtbloom::detail::gfn2::solve_eigensystems_cpu(
            evaluation.plan, evaluation.cache, 83u, hamiltonian.data(), 0.0, backend(),
            evaluation.scratch, evaluation.wavefunction, thermodynamics,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(evaluation.plan.alpha_electron_counts()[0] == 1.0);

  thermodynamics = evaluation.thermodynamics();
  thermodynamics.system_statuses = evaluation.cache.system_statuses;
  CHECK(xtbloom::detail::gfn2::solve_eigensystems_cpu(
            evaluation.plan, evaluation.cache, 83u, hamiltonian.data(), 0.0, backend(),
            evaluation.scratch, evaluation.wavefunction, thermodynamics,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  thermodynamics = evaluation.thermodynamics();
  CHECK(xtbloom::detail::gfn2::solve_eigensystems_cpu(
            evaluation.plan, evaluation.cache, 83u,
            reinterpret_cast<const double*>(&evaluation.scratch), 0.0, backend(),
            evaluation.scratch, evaluation.wavefunction, thermodynamics,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  AlignedBuffer shared_wavefunction_storage(
      std::max(evaluation.layout.workspace_size_bytes, evaluation.plan.workspace_size_bytes()));
  WavefunctionView shared_wavefunction;
  EigensolverWorkspace wavefunction_scratch;
  CHECK(xtbloom::detail::gfn2::bind_wavefunction_view(
            evaluation.layout, shared_wavefunction_storage.data, shared_wavefunction_storage.size,
            shared_wavefunction, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::bind_eigensolver_workspace(
            evaluation.plan, shared_wavefunction_storage.data, shared_wavefunction_storage.size,
            wavefunction_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::solve_eigensystems_cpu(
            evaluation.plan, evaluation.cache, 83u, hamiltonian.data(), 0.0, backend(),
            wavefunction_scratch, shared_wavefunction, evaluation.thermodynamics(),
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  return 0;
}

int test_negative_lapack_info_is_call_failure() {
  std::string error;
  Evaluation evaluation;
  CHECK(initialize_evaluation({0, 2}, {1, 1}, {0.0}, {0}, {1}, evaluation, error));
  CHECK(factor(evaluation, {1.0, 0.0, 0.0, 1.0}, 89u, error));
  fill_outputs(evaluation, 107.0);
  reset_backend_spies();
  dsyevd_failure_call.store(1, std::memory_order_relaxed);
  dsyevd_failure_info.store(-4, std::memory_order_relaxed);
  EigensolverThermodynamicsView thermodynamics = evaluation.thermodynamics();
  CHECK(xtbloom::detail::gfn2::solve_eigensystems_cpu(
            evaluation.plan, evaluation.cache, 89u,
            std::array<double, 4>{{-0.5, 0.0, 0.0, 0.2}}.data(), 0.0, backend(), evaluation.scratch,
            evaluation.wavefunction, thermodynamics, error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(evaluation.statuses[0] == 99);
  CHECK(evaluation.band_energies[0] == 107.0);
  reset_backend_spies();
  return 0;
}

int test_later_system_backend_failures_are_batch_atomic() {
  std::string error;
  Evaluation evaluation;
  CHECK(initialize_evaluation({0, 2, 4}, {1, 1, 1, 1}, {0.0, 0.0}, {0, 0}, {1, 1}, evaluation,
                              error));
  const std::vector<double> overlap{1.0, 0.1, 0.1, 1.0, 1.0, -0.2, -0.2, 1.0};

  std::fill_n(evaluation.cache.cholesky_factors,
              static_cast<std::size_t>(evaluation.plan.total_matrix_elements()), 109.0);
  std::fill_n(evaluation.cache.geometry_generations,
              static_cast<std::size_t>(evaluation.plan.batch_size()), 110u);
  std::fill_n(evaluation.cache.system_statuses,
              static_cast<std::size_t>(evaluation.plan.batch_size()),
              static_cast<xtbloom_status_t>(111));
  reset_backend_spies();
  dpotrf_failure_call.store(2, std::memory_order_relaxed);
  dpotrf_failure_info.store(-4, std::memory_order_relaxed);
  CHECK(xtbloom::detail::gfn2::factor_overlap_cpu(evaluation.plan, overlap.data(), 97u, backend(),
                                                  evaluation.scratch, evaluation.cache,
                                                  error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(potrf_calls.load(std::memory_order_relaxed) == 2);
  CHECK(dpocon_calls.load(std::memory_order_relaxed) == 1);
  CHECK(std::all_of(evaluation.cache.cholesky_factors,
                    evaluation.cache.cholesky_factors + evaluation.plan.total_matrix_elements(),
                    [](double value) { return value == 109.0; }));
  CHECK(std::all_of(evaluation.cache.geometry_generations,
                    evaluation.cache.geometry_generations + evaluation.plan.batch_size(),
                    [](std::uint64_t value) { return value == 110u; }));
  CHECK(std::all_of(evaluation.cache.system_statuses,
                    evaluation.cache.system_statuses + evaluation.plan.batch_size(),
                    [](xtbloom_status_t value) { return value == 111; }));

  std::fill_n(evaluation.cache.cholesky_factors,
              static_cast<std::size_t>(evaluation.plan.total_matrix_elements()), 113.0);
  std::fill_n(evaluation.cache.geometry_generations,
              static_cast<std::size_t>(evaluation.plan.batch_size()), 114u);
  std::fill_n(evaluation.cache.system_statuses,
              static_cast<std::size_t>(evaluation.plan.batch_size()),
              static_cast<xtbloom_status_t>(115));
  reset_backend_spies();
  dpocon_failure_call.store(2, std::memory_order_relaxed);
  dpocon_failure_info.store(-6, std::memory_order_relaxed);
  CHECK(xtbloom::detail::gfn2::factor_overlap_cpu(evaluation.plan, overlap.data(), 97u, backend(),
                                                  evaluation.scratch, evaluation.cache,
                                                  error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(potrf_calls.load(std::memory_order_relaxed) == 2);
  CHECK(dpocon_calls.load(std::memory_order_relaxed) == 2);
  CHECK(std::all_of(evaluation.cache.cholesky_factors,
                    evaluation.cache.cholesky_factors + evaluation.plan.total_matrix_elements(),
                    [](double value) { return value == 113.0; }));
  CHECK(std::all_of(evaluation.cache.geometry_generations,
                    evaluation.cache.geometry_generations + evaluation.plan.batch_size(),
                    [](std::uint64_t value) { return value == 114u; }));
  CHECK(std::all_of(evaluation.cache.system_statuses,
                    evaluation.cache.system_statuses + evaluation.plan.batch_size(),
                    [](xtbloom_status_t value) { return value == 115; }));

  reset_backend_spies();
  CHECK(factor(evaluation, overlap, 97u, error));
  fill_outputs(evaluation, 117.0);
  reset_backend_spies();
  dsyevd_failure_call.store(2, std::memory_order_relaxed);
  dsyevd_failure_info.store(-4, std::memory_order_relaxed);
  const std::vector<double> hamiltonian{-0.7, 0.03, 0.03, 0.2, -0.6, -0.04, -0.04, 0.3};
  EigensolverThermodynamicsView thermodynamics = evaluation.thermodynamics();
  CHECK(xtbloom::detail::gfn2::solve_eigensystems_cpu(
            evaluation.plan, evaluation.cache, 97u, hamiltonian.data(), 0.0, backend(),
            evaluation.scratch, evaluation.wavefunction, thermodynamics,
            error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(dsyevd_calls.load(std::memory_order_relaxed) == 2);
  CHECK(outputs_equal_sentinel(evaluation, 117.0));
  reset_backend_spies();
  return 0;
}

int test_second_spin_failure_is_atomic() {
  std::string error;
  Evaluation evaluation;
  CHECK(initialize_evaluation({0, 1}, {1}, {0.0}, {1}, {2}, evaluation, error));
  CHECK(factor(evaluation, {1.0}, 29u, error));
  fill_outputs(evaluation, 61.0);
  dsyevd_calls.store(0, std::memory_order_relaxed);
  dsyevd_failure_call.store(2, std::memory_order_relaxed);
  CHECK(solve(evaluation, {-0.5, -0.4}, 0.0, 29u, error));
  dsyevd_failure_call.store(0, std::memory_order_relaxed);
  CHECK(evaluation.statuses[0] == XTBLOOM_STATUS_EIGENSOLVER_FAILED);
  WavefunctionSystemView view;
  CHECK(system_view(evaluation, 0u, view, error));
  CHECK(view.coefficients[0] == 61.0 && view.coefficients[1] == 61.0);
  CHECK(view.eigenvalues[0] == 61.0 && view.eigenvalues[1] == 61.0);
  CHECK(view.density[0] == 61.0 && view.density[1] == 61.0);
  CHECK(evaluation.band_energies[0] == 61.0 && evaluation.free_energies[0] == 61.0);
  return 0;
}

int test_guarded_recycled_eigensolver_and_dense_fallback() {
  std::string error;
  constexpr std::size_t orbitals = 70u;
  std::vector<std::int32_t> atomic_numbers(orbitals, 1);
  std::vector<double> overlap(orbitals * orbitals, 0.0);
  std::vector<double> initial_hamiltonian(orbitals * orbitals, 0.0);
  for (std::size_t orbital = 0u; orbital < orbitals; ++orbital) {
    overlap[orbital * orbitals + orbital] = 1.0;
    initial_hamiltonian[orbital * orbitals + orbital] = -1.0 + 0.02 * static_cast<double>(orbital);
  }

  Evaluation recycled;
  Evaluation dense;
  CHECK(initialize_evaluation({0, static_cast<std::int64_t>(orbitals)}, atomic_numbers, {0.0}, {0},
                              {1}, recycled, error));
  CHECK(initialize_evaluation({0, static_cast<std::int64_t>(orbitals)}, atomic_numbers, {0.0}, {0},
                              {1}, dense, error));
  CHECK(factor(recycled, overlap, 101u, error));
  CHECK(factor(dense, overlap, 101u, error));
  CHECK(solve(recycled, initial_hamiltonian, 0.0, 101u, error));
  CHECK(solve(dense, initial_hamiltonian, 0.0, 101u, error));

  std::vector<double> block_hamiltonian = initial_hamiltonian;
  block_hamiltonian[4u * orbitals + 5u] = 2.0e-4;
  block_hamiltonian[5u * orbitals + 4u] = 2.0e-4;
  CHECK(solve(dense, block_hamiltonian, 0.0, 101u, error));
  EigensolverThermodynamicsView recycled_thermodynamics = recycled.thermodynamics();
  EigensolverSolveReport report;
  EigensolverRecyclePolicy recycle_policy;
  recycle_policy.minimum_orbitals = 64;
  reset_backend_spies();
  CHECK(xtbloom::detail::gfn2::solve_eigensystem_adaptive_cpu(
            recycled.plan, 0, recycled.cache, 101u, block_hamiltonian.data(), 0.0, backend(),
            recycled.scratch, recycled.wavefunction, recycled_thermodynamics, recycle_policy, true,
            report, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(error.empty());
  CHECK(report.mode == EigensolverSolveMode::kRecycled);
  CHECK(report.active_orbitals == 57);
  CHECK(report.maximum_backward_error < 1.0e-13);
  CHECK(report.rms_backward_error < 1.0e-13);
  CHECK(report.boundary_gap > 1.0e-3);
  CHECK(dsyevd_calls.load(std::memory_order_relaxed) == 2);
  WavefunctionSystemView recycled_view;
  WavefunctionSystemView dense_view;
  CHECK(system_view(recycled, 0u, recycled_view, error));
  CHECK(system_view(dense, 0u, dense_view, error));
  CHECK(generalized_eigensystem_is_valid(block_hamiltonian.data(), overlap.data(),
                                         recycled_view.coefficients, recycled_view.eigenvalues,
                                         orbitals));
  for (std::size_t element = 0u; element < orbitals * orbitals; ++element) {
    CHECK(near(recycled_view.density[element], dense_view.density[element], 2.0e-10));
    CHECK(near(recycled_view.energy_weighted_density[element],
               dense_view.energy_weighted_density[element], 2.0e-10));
  }
  CHECK(near(recycled.band_energies[0], dense.band_energies[0], 2.0e-10));

  /* Coupling the occupied block to an omitted virtual makes the exact
   * cross-block residual fail. The same logical iteration must complete with
   * the ordinary dense solve and must not publish an eigensolver failure. */
  std::vector<double> coupled_hamiltonian = block_hamiltonian;
  coupled_hamiltonian[34u * orbitals + 68u] = 0.05;
  coupled_hamiltonian[68u * orbitals + 34u] = 0.05;
  reset_backend_spies();
  CHECK(xtbloom::detail::gfn2::solve_eigensystem_adaptive_cpu(
            recycled.plan, 0, recycled.cache, 101u, coupled_hamiltonian.data(), 0.0, backend(),
            recycled.scratch, recycled.wavefunction, recycled_thermodynamics, recycle_policy, true,
            report, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(report.mode == EigensolverSolveMode::kRecycleFallback);
  CHECK(recycled.statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(dsyevd_calls.load(std::memory_order_relaxed) == 3);
  CHECK(system_view(recycled, 0u, recycled_view, error));
  CHECK(generalized_eigensystem_is_valid(coupled_hamiltonian.data(), overlap.data(),
                                         recycled_view.coefficients, recycled_view.eigenvalues,
                                         orbitals));

  /* A previous frontier that is too close to the retained-complement
   * boundary must reject before either partial solve. The ordinary dense
   * solve still completes the same logical iteration. */
  Evaluation near_degenerate;
  CHECK(initialize_evaluation({0, static_cast<std::int64_t>(orbitals)}, atomic_numbers, {0.0}, {0},
                              {1}, near_degenerate, error));
  CHECK(factor(near_degenerate, overlap, 101u, error));
  std::vector<double> near_degenerate_hamiltonian = initial_hamiltonian;
  constexpr std::size_t boundary = 49u;
  near_degenerate_hamiltonian[boundary * orbitals + boundary] =
      near_degenerate_hamiltonian[(boundary - 1u) * orbitals + boundary - 1u] + 5.0e-7;
  CHECK(solve(near_degenerate, near_degenerate_hamiltonian, 0.0, 101u, error));
  EigensolverThermodynamicsView near_degenerate_thermodynamics = near_degenerate.thermodynamics();
  reset_backend_spies();
  CHECK(xtbloom::detail::gfn2::solve_eigensystem_adaptive_cpu(
            near_degenerate.plan, 0, near_degenerate.cache, 101u,
            near_degenerate_hamiltonian.data(), 0.0, backend(), near_degenerate.scratch,
            near_degenerate.wavefunction, near_degenerate_thermodynamics, recycle_policy, true,
            report, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(report.mode == EigensolverSolveMode::kRecycleFallback);
  CHECK(dsyevd_calls.load(std::memory_order_relaxed) == 1);

  /* A finite-temperature occupation tail extending into the omitted block
   * invalidates the subspace truncation even when the Hamiltonian itself is
   * unchanged. This guard is evaluated before any partial eigensolve. */
  Evaluation finite_temperature;
  CHECK(initialize_evaluation({0, static_cast<std::int64_t>(orbitals)}, atomic_numbers, {0.0}, {0},
                              {1}, finite_temperature, error));
  CHECK(factor(finite_temperature, overlap, 101u, error));
  constexpr double broad_temperature = 5.0e-2;
  CHECK(solve(finite_temperature, initial_hamiltonian, broad_temperature, 101u, error));
  EigensolverThermodynamicsView finite_temperature_thermodynamics =
      finite_temperature.thermodynamics();
  reset_backend_spies();
  CHECK(xtbloom::detail::gfn2::solve_eigensystem_adaptive_cpu(
            finite_temperature.plan, 0, finite_temperature.cache, 101u, initial_hamiltonian.data(),
            broad_temperature, backend(), finite_temperature.scratch,
            finite_temperature.wavefunction, finite_temperature_thermodynamics, recycle_policy,
            true, report, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(report.mode == EigensolverSolveMode::kRecycleFallback);
  CHECK(dsyevd_calls.load(std::memory_order_relaxed) == 1);

  /* Unrestricted systems retain the existing two-channel dense semantics. */
  Evaluation unrestricted;
  CHECK(initialize_evaluation({0, 4}, {1, 1, 1, 1}, {0.0}, {0}, {2}, unrestricted, error));
  const std::vector<double> small_overlap{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                          0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  const std::vector<double> small_hamiltonian{
      -0.8, 0.0, 0.0, 0.0, 0.0, -0.4, 0.0, 0.0, 0.0, 0.0, 0.2,  0.0, 0.0, 0.0, 0.0, 0.6,
      -0.7, 0.0, 0.0, 0.0, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.65};
  CHECK(factor(unrestricted, small_overlap, 103u, error));
  CHECK(solve(unrestricted, small_hamiltonian, 0.0, 103u, error));
  EigensolverRecyclePolicy unrestricted_policy;
  unrestricted_policy.minimum_orbitals = 1;
  unrestricted_policy.minimum_virtual_buffer = 1;
  unrestricted_policy.virtual_buffer_fraction = 0.0;
  EigensolverThermodynamicsView unrestricted_thermodynamics = unrestricted.thermodynamics();
  reset_backend_spies();
  CHECK(xtbloom::detail::gfn2::solve_eigensystem_adaptive_cpu(
            unrestricted.plan, 0, unrestricted.cache, 103u, small_hamiltonian.data(), 0.0,
            backend(), unrestricted.scratch, unrestricted.wavefunction, unrestricted_thermodynamics,
            unrestricted_policy, true, report, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(report.mode == EigensolverSolveMode::kRecycleFallback);
  CHECK(dsyevd_calls.load(std::memory_order_relaxed) == 2);

  EigensolverRecyclePolicy invalid_policy;
  invalid_policy.minimum_orbitals = 0;
  const EigensolverSolveReport saved_report = report;
  CHECK(xtbloom::detail::gfn2::solve_eigensystem_adaptive_cpu(
            unrestricted.plan, 0, unrestricted.cache, 103u, small_hamiltonian.data(), 0.0,
            backend(), unrestricted.scratch, unrestricted.wavefunction, unrestricted_thermodynamics,
            invalid_policy, true, report, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(report.mode == saved_report.mode &&
        report.active_orbitals == saved_report.active_orbitals &&
        report.maximum_backward_error == saved_report.maximum_backward_error &&
        report.rms_backward_error == saved_report.rms_backward_error &&
        report.boundary_gap == saved_report.boundary_gap &&
        report.residual_gap_ratio == saved_report.residual_gap_ratio);
  return 0;
}

int test_batch_matches_sequential() {
  std::string error;
  Evaluation batch;
  CHECK(initialize_evaluation({0, 2, 3}, {1, 1, 1}, {0.0, 0.0}, {0, 1}, {1, 2}, batch, error));
  const std::vector<double> batch_overlap{1.0, 0.1, 0.1, 1.0, 1.0};
  const std::vector<double> batch_hamiltonian{-0.6, 0.04, 0.04, 0.2, -0.5, -0.4};
  CHECK(factor(batch, batch_overlap, 37u, error));
  fill_outputs(batch, 71.0);
  CHECK(solve(batch, batch_hamiltonian, XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE, 37u, error));

  const std::vector<double> expected_coefficients(
      batch.wavefunction.coefficients,
      batch.wavefunction.coefficients + batch.layout.coefficients.element_count);
  const std::vector<double> expected_eigenvalues(
      batch.wavefunction.eigenvalues,
      batch.wavefunction.eigenvalues + batch.layout.eigenvalues.element_count);
  const std::vector<double> expected_occupations(
      batch.wavefunction.occupations,
      batch.wavefunction.occupations + batch.layout.occupations.element_count);
  const std::vector<double> expected_densities(
      batch.wavefunction.density, batch.wavefunction.density + batch.layout.density.element_count);
  const std::vector<double> expected_weighted_densities(
      batch.wavefunction.energy_weighted_density,
      batch.wavefunction.energy_weighted_density +
          batch.layout.energy_weighted_density.element_count);
  const auto expected_mu = batch.chemical_potentials;
  const auto expected_entropy = batch.entropies;
  const auto expected_band = batch.band_energies;
  const auto expected_free = batch.free_energies;

  CHECK(batch.plan.worker_workspace_size_bytes() < batch.plan.workspace_size_bytes());
  AlignedBuffer first_worker_storage(batch.plan.worker_workspace_size_bytes());
  AlignedBuffer second_worker_storage(batch.plan.worker_workspace_size_bytes());
  EigensolverWorkspace first_worker;
  EigensolverWorkspace second_worker;
  CHECK(xtbloom::detail::gfn2::bind_eigensolver_worker_workspace(
            batch.plan, first_worker_storage.data, first_worker_storage.size, first_worker,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::bind_eigensolver_worker_workspace(
            batch.plan, second_worker_storage.data, second_worker_storage.size, second_worker,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(first_worker.factor_staging == nullptr && first_worker.batch_coefficients == nullptr);
  CHECK(second_worker.factor_staging == nullptr && second_worker.batch_coefficients == nullptr);

  auto* const staging_begin = static_cast<std::byte*>(batch.scratch.workspace_base) +
                              batch.plan.worker_workspace_size_bytes();
  auto* const staging_end =
      static_cast<std::byte*>(batch.scratch.workspace_base) + batch.plan.workspace_size_bytes();
  std::fill(staging_begin, staging_end, std::byte{0xA5});
  const std::vector<std::byte> staging_snapshot(staging_begin, staging_end);

  fill_outputs(batch, 72.0);
  EigensolverThermodynamicsView first_worker_thermodynamics = batch.thermodynamics();
  EigensolverThermodynamicsView second_worker_thermodynamics = batch.thermodynamics();
  std::string first_worker_error;
  std::string second_worker_error;
  xtbloom_status_t first_thread_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  xtbloom_status_t second_thread_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  std::thread first_thread([&] {
    first_thread_status = xtbloom::detail::gfn2::solve_eigensystem_cpu(
        batch.plan, 0, batch.cache, 37u, batch_hamiltonian.data(),
        XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE, backend(), first_worker, batch.wavefunction,
        first_worker_thermodynamics, first_worker_error);
  });
  std::thread second_thread([&] {
    second_thread_status = xtbloom::detail::gfn2::solve_eigensystem_cpu(
        batch.plan, 1, batch.cache, 37u, batch_hamiltonian.data() + 4u,
        XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE, backend(), second_worker, batch.wavefunction,
        second_worker_thermodynamics, second_worker_error);
  });
  first_thread.join();
  second_thread.join();
  CHECK(first_thread_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(second_thread_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(first_worker_error.empty() && second_worker_error.empty());
  CHECK(std::equal(expected_coefficients.begin(), expected_coefficients.end(),
                   batch.wavefunction.coefficients));
  CHECK(std::equal(expected_eigenvalues.begin(), expected_eigenvalues.end(),
                   batch.wavefunction.eigenvalues));
  CHECK(std::equal(expected_occupations.begin(), expected_occupations.end(),
                   batch.wavefunction.occupations));
  CHECK(
      std::equal(expected_densities.begin(), expected_densities.end(), batch.wavefunction.density));
  CHECK(std::equal(expected_weighted_densities.begin(), expected_weighted_densities.end(),
                   batch.wavefunction.energy_weighted_density));
  CHECK(batch.chemical_potentials == expected_mu);
  CHECK(batch.entropies == expected_entropy);
  CHECK(batch.band_energies == expected_band);
  CHECK(batch.free_energies == expected_free);
  CHECK(std::equal(staging_snapshot.begin(), staging_snapshot.end(), staging_begin));

  allocation_test::new_count.store(0u, std::memory_order_relaxed);
  allocation_test::malloc_count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const xtbloom_status_t first_worker_status = xtbloom::detail::gfn2::solve_eigensystem_cpu(
      batch.plan, 0, batch.cache, 37u, batch_hamiltonian.data(),
      XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE, backend(), first_worker, batch.wavefunction,
      first_worker_thermodynamics, first_worker_error);
  const xtbloom_status_t second_worker_status = xtbloom::detail::gfn2::solve_eigensystem_cpu(
      batch.plan, 1, batch.cache, 37u, batch_hamiltonian.data() + 4u,
      XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE, backend(), second_worker, batch.wavefunction,
      second_worker_thermodynamics, second_worker_error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  CHECK(first_worker_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(second_worker_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(allocation_test::new_count.load(std::memory_order_relaxed) == 0u);
#if !defined(XTBLOOM_EIGENSOLVER_TEST_ASAN) && defined(__GLIBC__)
  CHECK(allocation_test::malloc_count.load(std::memory_order_relaxed) == 0u);
#endif
  CHECK(std::equal(staging_snapshot.begin(), staging_snapshot.end(), staging_begin));

  Evaluation first;
  Evaluation second;
  Evaluation expanded_batch;
  CHECK(initialize_evaluation({0, 2}, {1, 1}, {0.0}, {0}, {1}, first, error));
  CHECK(initialize_evaluation({0, 1}, {1}, {0.0}, {1}, {2}, second, error));
  CHECK(initialize_evaluation({0, 2, 4, 6, 8}, {1, 1, 1, 1, 1, 1, 1, 1}, {0.0, 0.0, 0.0, 0.0},
                              {0, 0, 0, 0}, {1, 1, 1, 1}, expanded_batch, error));
  CHECK(first.plan.maximum_orbitals() == batch.plan.maximum_orbitals());
  CHECK(first.plan.worker_workspace_size_bytes() == batch.plan.worker_workspace_size_bytes());
  CHECK(first.plan.maximum_orbitals() == expanded_batch.plan.maximum_orbitals());
  CHECK(first.plan.worker_workspace_size_bytes() ==
        expanded_batch.plan.worker_workspace_size_bytes());
  CHECK(first.plan.workspace_size_bytes() < expanded_batch.plan.workspace_size_bytes());
  CHECK(factor(first, {1.0, 0.1, 0.1, 1.0}, 37u, error));
  CHECK(factor(second, {1.0}, 37u, error));
  fill_outputs(first, 71.0);
  fill_outputs(second, 71.0);
  CHECK(solve(first, {-0.6, 0.04, 0.04, 0.2}, XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE, 37u, error));
  CHECK(solve(second, {-0.5, -0.4}, XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE, 37u, error));

  WavefunctionSystemView batch_first;
  WavefunctionSystemView batch_second;
  WavefunctionSystemView sequential_first;
  WavefunctionSystemView sequential_second;
  CHECK(system_view(batch, 0u, batch_first, error));
  CHECK(system_view(batch, 1u, batch_second, error));
  CHECK(system_view(first, 0u, sequential_first, error));
  CHECK(system_view(second, 0u, sequential_second, error));
  for (std::size_t element = 0u; element < 4u; ++element) {
    CHECK(near(batch_first.density[element], sequential_first.density[element]));
    CHECK(near(batch_first.energy_weighted_density[element],
               sequential_first.energy_weighted_density[element]));
  }
  for (std::size_t element = 0u; element < 2u; ++element) {
    CHECK(near(batch_second.density[element], sequential_second.density[element]));
    CHECK(near(batch_second.energy_weighted_density[element],
               sequential_second.energy_weighted_density[element]));
  }
  CHECK(near(batch.band_energies[0], first.band_energies[0]));
  CHECK(near(batch.free_energies[0], first.free_energies[0]));
  CHECK(near(batch.band_energies[1], second.band_energies[0]));
  CHECK(near(batch.free_energies[1], second.free_energies[0]));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--mkl-ilp64-rejection-child") == 0) {
    return run_mkl_ilp64_rejection_child();
  }
  /* Complete the injected backend's numerical preflight before call-count tests. */
  static_cast<void>(backend());
  if (const int status = test_occupations_degeneracy_and_fractional_filling(); status != 0) {
    return status;
  }
  if (const int status = test_binary64_frontier_retry_controls(); status != 0) {
    return status;
  }
  if (const int status = test_occupation_representability_policy(); status != 0) {
    return status;
  }
  if (const int status = test_mkl_ilp64_rejection_in_fresh_process(); status != 0) {
    return status;
  }
  if (const int status = test_production_lp64_factory(); status != 0) {
    return status;
  }
  if (const int status = test_unrestricted_literal_generalized_eigenproblem(); status != 0) {
    return status;
  }
  if (const int status = test_tblite_300_kelvin_pinned_oracle(); status != 0) {
    return status;
  }
  if (const int status = test_restricted_closed_shell_and_warm_reuse(); status != 0) {
    return status;
  }
  if (const int status = test_open_shell_restricted_radical_and_spin_singlet(); status != 0) {
    return status;
  }
  if (const int status = test_finite_temperature_free_energy(); status != 0) {
    return status;
  }
  if (const int status = test_ragged_failure_and_atomicity(); status != 0) {
    return status;
  }
  if (const int status = test_plan_identity_generation_and_alias_rejection(); status != 0) {
    return status;
  }
  if (const int status = test_negative_lapack_info_is_call_failure(); status != 0) {
    return status;
  }
  if (const int status = test_later_system_backend_failures_are_batch_atomic(); status != 0) {
    return status;
  }
  if (const int status = test_second_spin_failure_is_atomic(); status != 0) {
    return status;
  }
  if (const int status = test_guarded_recycled_eigensolver_and_dense_fallback(); status != 0) {
    return status;
  }
  if (const int status = test_batch_matches_sequential(); status != 0) {
    return status;
  }
  return 0;
}
