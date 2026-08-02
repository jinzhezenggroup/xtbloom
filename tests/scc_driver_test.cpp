#include "model/gfn2/scc_driver.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "model/gfn2/basis.hpp"
#include "model/gfn2/h0.hpp"
#include "model/gfn2/integrals.hpp"

namespace allocation_test {
std::atomic<std::size_t> count{0u};
std::atomic<bool> enabled{false};
}  // namespace allocation_test

void* operator new(std::size_t size) {
  if (allocation_test::enabled.load(std::memory_order_relaxed)) {
    allocation_test::count.fetch_add(1u, std::memory_order_relaxed);
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

using namespace gpuxtb::detail::gfn2;

struct AlignedBuffer {
  void* data = nullptr;
  std::size_t size = 0u;

  explicit AlignedBuffer(std::size_t requested) {
    size = (std::max<std::size_t>(requested, 1u) + 63u) & ~std::size_t{63u};
    data = std::aligned_alloc(64u, size);
    if (data != nullptr) {
      std::memset(data, 0, size);
    }
  }
  ~AlignedBuffer() { std::free(data); }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
};

std::atomic<int> diagonalizations{0};

LapackInt tiny_dpotrf(LapackInt, char, LapackInt n, double* matrix, LapackInt) {
  if (n != 1 || !(matrix[0] > 0.0)) {
    return 1;
  }
  matrix[0] = std::sqrt(matrix[0]);
  return 0;
}

LapackInt tiny_dpocon(LapackInt, char, LapackInt n, const double*, LapackInt, double,
                      double* reciprocal_condition, double*, LapackInt*) {
  if (n != 1) {
    return -3;
  }
  *reciprocal_condition = 1.0;
  return 0;
}

LapackInt tiny_dsyevd(LapackInt, char, char, LapackInt n, double* matrix, LapackInt,
                      double* eigenvalues, double*, LapackInt, LapackInt*, LapackInt) {
  if (n != 1) {
    return -4;
  }
  diagonalizations.fetch_add(1, std::memory_order_relaxed);
  eigenvalues[0] = matrix[0];
  matrix[0] = 1.0;
  return 0;
}

void tiny_dtrsm(int, int, int, int, int, LapackInt rows, LapackInt columns, double alpha,
                const double* triangular, LapackInt, double* rhs, LapackInt) {
  for (LapackInt column = 0; column < columns; ++column) {
    for (LapackInt row = 0; row < rows; ++row) {
      rhs[column * rows + row] *= alpha / triangular[0];
    }
  }
}

void tiny_dgemm(int, int, int, LapackInt rows, LapackInt columns, LapackInt inner, double alpha,
                const double* left, LapackInt leading_left, const double* right,
                LapackInt leading_right, double beta, double* result, LapackInt leading_result) {
  for (LapackInt column = 0; column < columns; ++column) {
    for (LapackInt row = 0; row < rows; ++row) {
      double value = 0.0;
      for (LapackInt k = 0; k < inner; ++k) {
        value += left[k * leading_left + row] * right[k * leading_right + column];
      }
      result[column * leading_result + row] =
          alpha * value + beta * result[column * leading_result + row];
    }
  }
}

const CpuLinearAlgebraBackend& backend() {
  static const CpuLinearAlgebraBackend value = [] {
    CpuLinearAlgebraBackend result;
    std::string error;
    if (make_internal_test_lp64_backend(&tiny_dpotrf, &tiny_dpocon, &tiny_dsyevd, &tiny_dtrsm,
                                        &tiny_dgemm, nullptr, result,
                                        error) != GPUXTB_STATUS_SUCCESS) {
      std::abort();
    }
    return result;
  }();
  return value;
}

struct Fixture {
  std::int64_t batch_size = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> charges;
  std::vector<std::int32_t> unpaired;
  std::vector<std::int32_t> spins;
  std::vector<double> coordination;

  BasisPlan basis;
  IntegralPlan integral_plan;
  H0Plan h0_plan;
  WavefunctionLayout wavefunction_layout;
  ES2Plan es2_plan;
  ES3Plan es3_plan;
  AES2Plan aes2_plan;
  MullikenPlan mulliken_plan;
  EigensolverPlan eigensolver_plan;
  SccMixerPlan mixer_plan;
  SccDriverPlan driver_plan;

  std::vector<double> overlap;
  std::vector<double> dipole_integrals;
  std::vector<double> quadrupole_integrals;
  std::vector<double> h0;
  std::unique_ptr<AlignedBuffer> integral_scratch;

  std::unique_ptr<AlignedBuffer> es2_storage;
  std::unique_ptr<AlignedBuffer> es2_scratch_storage;
  ES2GeometryCache es2_cache;
  ES2Workspace es2_scratch;
  std::unique_ptr<AlignedBuffer> aes2_storage;
  std::unique_ptr<AlignedBuffer> aes2_scratch_storage;
  AES2GeometryCache aes2_cache;
  AES2Workspace aes2_scratch;

  std::unique_ptr<AlignedBuffer> wavefunction_storage;
  WavefunctionView wavefunction;
  std::unique_ptr<AlignedBuffer> overlap_cache_storage;
  EigensolverOverlapCache overlap_cache;
  std::unique_ptr<AlignedBuffer> eigensolver_scratch_storage;
  EigensolverWorkspace eigensolver_scratch;
  std::unique_ptr<AlignedBuffer> mixer_state_storage;
  SccMixerState mixer_state;
  std::unique_ptr<AlignedBuffer> driver_state_storage;
  SccDriverState driver_state;
  std::unique_ptr<AlignedBuffer> driver_scratch_storage;
  SccDriverWorkspace driver_scratch;
  SccDriverGeometryView geometry;
};

struct ComponentPlans {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int32_t> spin_channels;
  BasisPlan basis;
  IntegralPlan integrals;
  WavefunctionLayout wavefunction;
  MullikenPlan mulliken;
  ES2Plan es2;
  ES3Plan es3;
  AES2Plan aes2;
  EigensolverPlan eigensolver;
  SccMixerPlan mixer;
};

bool make_component_plans(std::vector<std::int32_t> atomic_numbers, double molecular_charge,
                          std::int32_t unpaired_electrons, ComponentPlans& plans,
                          std::string& error) {
  plans.atomic_numbers = std::move(atomic_numbers);
  plans.atom_offsets = {0, static_cast<std::int64_t>(plans.atomic_numbers.size())};
  plans.molecular_charges = {molecular_charge};
  plans.unpaired_electrons = {unpaired_electrons};
  plans.spin_channels = {1};
  return make_basis_plan(1, plans.atom_offsets.back(), plans.atom_offsets.data(),
                         plans.atomic_numbers.data(), plans.basis,
                         error) == GPUXTB_STATUS_SUCCESS &&
         make_integral_plan(plans.basis, plans.integrals, error) == GPUXTB_STATUS_SUCCESS &&
         make_wavefunction_layout(plans.basis, plans.atomic_numbers.data(),
                                  plans.molecular_charges.data(), plans.unpaired_electrons.data(),
                                  plans.spin_channels.data(), plans.wavefunction,
                                  error) == GPUXTB_STATUS_SUCCESS &&
         make_mulliken_plan(plans.basis, plans.integrals, plans.wavefunction, plans.mulliken,
                            error) == GPUXTB_STATUS_SUCCESS &&
         make_es2_plan(plans.basis, plans.atomic_numbers.data(), plans.es2, error) ==
             GPUXTB_STATUS_SUCCESS &&
         make_es3_plan(plans.basis, plans.atomic_numbers.data(), plans.es3, error) ==
             GPUXTB_STATUS_SUCCESS &&
         make_aes2_plan(plans.basis, plans.atomic_numbers.data(), plans.aes2, error) ==
             GPUXTB_STATUS_SUCCESS &&
         make_eigensolver_plan(plans.wavefunction, plans.eigensolver, error) ==
             GPUXTB_STATUS_SUCCESS &&
         make_scc_mixer_plan(plans.wavefunction, 3, 0.4, 1.0e-10, 1.0e-10, plans.mixer, error) ==
             GPUXTB_STATUS_SUCCESS;
}

bool make_fixture(std::int64_t batch_size, Fixture& fixture, std::string& error,
                  std::uint64_t maximum_iterations = 5u, double mixer_tolerance = 1.0e-10) {
  fixture.batch_size = batch_size;
  fixture.atom_offsets.resize(static_cast<std::size_t>(batch_size) + 1u);
  for (std::int64_t system = 0; system <= batch_size; ++system) {
    fixture.atom_offsets[static_cast<std::size_t>(system)] = system;
  }
  fixture.atomic_numbers.assign(static_cast<std::size_t>(batch_size), 1);
  fixture.positions.assign(static_cast<std::size_t>(3 * batch_size), 0.0);
  fixture.charges.assign(static_cast<std::size_t>(batch_size), 1.0);  // isolated H+
  fixture.unpaired.assign(static_cast<std::size_t>(batch_size), 0);
  fixture.spins.assign(static_cast<std::size_t>(batch_size), 1);
  fixture.coordination.assign(static_cast<std::size_t>(batch_size), 0.0);

  if (make_basis_plan(batch_size, batch_size, fixture.atom_offsets.data(),
                      fixture.atomic_numbers.data(), fixture.basis,
                      error) != GPUXTB_STATUS_SUCCESS ||
      make_integral_plan(fixture.basis, fixture.integral_plan, error) != GPUXTB_STATUS_SUCCESS ||
      make_h0_plan(fixture.basis, fixture.integral_plan, fixture.atomic_numbers.data(),
                   fixture.h0_plan, error) != GPUXTB_STATUS_SUCCESS ||
      make_wavefunction_layout(fixture.basis, fixture.atomic_numbers.data(), fixture.charges.data(),
                               fixture.unpaired.data(), fixture.spins.data(),
                               fixture.wavefunction_layout, error) != GPUXTB_STATUS_SUCCESS ||
      make_es2_plan(fixture.basis, fixture.atomic_numbers.data(), fixture.es2_plan, error) !=
          GPUXTB_STATUS_SUCCESS ||
      make_es3_plan(fixture.basis, fixture.atomic_numbers.data(), fixture.es3_plan, error) !=
          GPUXTB_STATUS_SUCCESS ||
      make_aes2_plan(fixture.basis, fixture.atomic_numbers.data(), fixture.aes2_plan, error) !=
          GPUXTB_STATUS_SUCCESS ||
      make_mulliken_plan(fixture.basis, fixture.integral_plan, fixture.wavefunction_layout,
                         fixture.mulliken_plan, error) != GPUXTB_STATUS_SUCCESS ||
      make_eigensolver_plan(fixture.wavefunction_layout, fixture.eigensolver_plan, error) !=
          GPUXTB_STATUS_SUCCESS ||
      make_scc_mixer_plan(fixture.wavefunction_layout, 3, 0.4, mixer_tolerance, mixer_tolerance,
                          fixture.mixer_plan, error) != GPUXTB_STATUS_SUCCESS ||
      make_scc_driver_plan(fixture.wavefunction_layout, fixture.mulliken_plan, fixture.es2_plan,
                           fixture.es3_plan, fixture.aes2_plan, fixture.eigensolver_plan,
                           fixture.mixer_plan, maximum_iterations, 0.0, fixture.driver_plan,
                           error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }

  fixture.overlap.resize(static_cast<std::size_t>(fixture.integral_plan.total_matrix_elements));
  fixture.dipole_integrals.resize(
      static_cast<std::size_t>(3 * fixture.integral_plan.total_matrix_elements));
  fixture.quadrupole_integrals.resize(
      static_cast<std::size_t>(6 * fixture.integral_plan.total_matrix_elements));
  fixture.h0.resize(static_cast<std::size_t>(fixture.integral_plan.total_matrix_elements));
  fixture.integral_scratch =
      std::make_unique<AlignedBuffer>(fixture.integral_plan.workspace_size_bytes);
  if (evaluate_overlap_cpu(fixture.basis, fixture.integral_plan, fixture.positions.data(),
                           fixture.overlap.data(), fixture.integral_scratch->data,
                           fixture.integral_scratch->size, error) != GPUXTB_STATUS_SUCCESS ||
      evaluate_multipole_cpu(fixture.basis, fixture.integral_plan, fixture.positions.data(),
                             fixture.dipole_integrals.data(), fixture.quadrupole_integrals.data(),
                             fixture.integral_scratch->data, fixture.integral_scratch->size,
                             error) != GPUXTB_STATUS_SUCCESS ||
      evaluate_h0_cpu(fixture.basis, fixture.integral_plan, fixture.h0_plan,
                      fixture.positions.data(), fixture.coordination.data(), fixture.overlap.data(),
                      fixture.h0.data(), error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }

  fixture.es2_storage = std::make_unique<AlignedBuffer>(
      static_cast<std::size_t>(fixture.es2_plan.total_matrix_elements()) * sizeof(double));
  fixture.es2_scratch_storage = std::make_unique<AlignedBuffer>(
      static_cast<std::size_t>(fixture.es2_plan.total_matrix_elements()) * sizeof(double));
  fixture.es2_scratch.matrix_scratch = static_cast<double*>(fixture.es2_scratch_storage->data);
  fixture.es2_scratch.matrix_elements = fixture.es2_plan.total_matrix_elements();
  if (update_es2_geometry_cache_cpu(
          fixture.es2_plan, fixture.positions.data(), 1u,
          static_cast<double*>(fixture.es2_storage->data),
          static_cast<std::size_t>(fixture.es2_plan.total_matrix_elements()), fixture.es2_scratch,
          fixture.es2_cache, error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }

  fixture.aes2_storage = std::make_unique<AlignedBuffer>(
      static_cast<std::size_t>(fixture.aes2_plan.pair_data_elements()) * sizeof(double));
  fixture.aes2_scratch_storage = std::make_unique<AlignedBuffer>(
      static_cast<std::size_t>(fixture.aes2_plan.pair_data_elements()) * sizeof(double));
  fixture.aes2_scratch.pair_scratch = static_cast<double*>(fixture.aes2_scratch_storage->data);
  fixture.aes2_scratch.pair_elements = fixture.aes2_plan.pair_data_elements();
  if (update_aes2_geometry_cache_cpu(
          fixture.aes2_plan, fixture.positions.data(), fixture.coordination.data(), 1u,
          static_cast<double*>(fixture.aes2_storage->data),
          static_cast<std::size_t>(fixture.aes2_plan.pair_data_elements()), fixture.aes2_scratch,
          fixture.aes2_cache, error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }

  fixture.wavefunction_storage =
      std::make_unique<AlignedBuffer>(fixture.wavefunction_layout.workspace_size_bytes);
  fixture.overlap_cache_storage =
      std::make_unique<AlignedBuffer>(fixture.eigensolver_plan.overlap_cache_size_bytes());
  fixture.eigensolver_scratch_storage =
      std::make_unique<AlignedBuffer>(fixture.eigensolver_plan.workspace_size_bytes());
  fixture.mixer_state_storage =
      std::make_unique<AlignedBuffer>(fixture.mixer_plan.state_size_bytes());
  fixture.driver_state_storage =
      std::make_unique<AlignedBuffer>(fixture.driver_plan.state_size_bytes());
  fixture.driver_scratch_storage =
      std::make_unique<AlignedBuffer>(fixture.driver_plan.workspace_size_bytes());
  if (bind_wavefunction_view(fixture.wavefunction_layout, fixture.wavefunction_storage->data,
                             fixture.wavefunction_storage->size, fixture.wavefunction,
                             error) != GPUXTB_STATUS_SUCCESS ||
      initialize_sad_multipole_state(fixture.wavefunction_layout, fixture.wavefunction, error) !=
          GPUXTB_STATUS_SUCCESS ||
      bind_eigensolver_overlap_cache(fixture.eigensolver_plan, fixture.overlap_cache_storage->data,
                                     fixture.overlap_cache_storage->size, fixture.overlap_cache,
                                     error) != GPUXTB_STATUS_SUCCESS ||
      bind_eigensolver_workspace(fixture.eigensolver_plan,
                                 fixture.eigensolver_scratch_storage->data,
                                 fixture.eigensolver_scratch_storage->size,
                                 fixture.eigensolver_scratch, error) != GPUXTB_STATUS_SUCCESS ||
      factor_overlap_cpu(fixture.eigensolver_plan, fixture.overlap.data(), 1u, backend(),
                         fixture.eigensolver_scratch, fixture.overlap_cache,
                         error) != GPUXTB_STATUS_SUCCESS ||
      bind_scc_mixer_state(fixture.mixer_plan, fixture.mixer_state_storage->data,
                           fixture.mixer_state_storage->size, fixture.mixer_state,
                           error) != GPUXTB_STATUS_SUCCESS ||
      bind_scc_driver_state(fixture.driver_plan, fixture.driver_state_storage->data,
                            fixture.driver_state_storage->size, fixture.driver_state,
                            error) != GPUXTB_STATUS_SUCCESS ||
      bind_scc_driver_workspace(fixture.driver_plan, fixture.driver_scratch_storage->data,
                                fixture.driver_scratch_storage->size, fixture.driver_scratch,
                                error) != GPUXTB_STATUS_SUCCESS ||
      initialize_scc_driver_state_cpu(fixture.driver_plan, fixture.wavefunction,
                                      fixture.mixer_state, fixture.driver_state,
                                      error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }

  fixture.geometry.h0 = fixture.h0.data();
  fixture.geometry.h0_elements = fixture.integral_plan.total_matrix_elements;
  fixture.geometry.integrals = {
      fixture.overlap.data(), fixture.dipole_integrals.data(), fixture.quadrupole_integrals.data(),
      fixture.integral_plan.total_matrix_elements, fixture.mulliken_plan.identity()};
  fixture.geometry.es2_cache = fixture.es2_cache;
  fixture.geometry.aes2_cache = fixture.aes2_cache;
  fixture.geometry.geometry_generation = 1u;
  return true;
}

int test_ragged_failure_isolation_restart_and_skip() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(2, fixture, error));
  fixture.overlap_cache.system_statuses[1] = GPUXTB_STATUS_EIGENSOLVER_FAILED;
  diagonalizations.store(0, std::memory_order_relaxed);
  CHECK(iterate_scc_driver_batch_cpu(
            fixture.driver_plan, fixture.geometry, backend(), fixture.overlap_cache,
            fixture.wavefunction, fixture.mixer_state, fixture.driver_state, fixture.driver_scratch,
            error) == GPUXTB_STATUS_EIGENSOLVER_FAILED);
  CHECK(diagonalizations.load(std::memory_order_relaxed) == 1);
  CHECK(fixture.driver_state.system_statuses[0] == GPUXTB_STATUS_SUCCESS);
  CHECK(fixture.driver_state.converged[0] == 1u);
  CHECK(fixture.driver_state.iterations[0] == 1u);
  CHECK(fixture.driver_state.system_statuses[1] == GPUXTB_STATUS_EIGENSOLVER_FAILED);
  CHECK(fixture.driver_state.iterations[1] == 1u);
  CHECK(std::isnan(fixture.driver_state.free_energies[1]));

  const double first_qsh = fixture.wavefunction.qsh[0];
  const int calls_before_skip = diagonalizations.load(std::memory_order_relaxed);
  CHECK(iterate_scc_driver_batch_cpu(fixture.driver_plan, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state,
                                     fixture.driver_scratch, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(diagonalizations.load(std::memory_order_relaxed) == calls_before_skip);
  CHECK(fixture.wavefunction.qsh[0] == first_qsh);

  fixture.overlap_cache.system_statuses[1] = GPUXTB_STATUS_SUCCESS;
  CHECK(restart_scc_driver_system_cpu(fixture.driver_plan, 1, fixture.wavefunction,
                                      fixture.mixer_state, fixture.driver_state,
                                      error) == GPUXTB_STATUS_SUCCESS);
  CHECK(iterate_scc_driver_batch_cpu(fixture.driver_plan, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state,
                                     fixture.driver_scratch, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(diagonalizations.load(std::memory_order_relaxed) == calls_before_skip + 1);
  CHECK(fixture.driver_state.converged[0] == 1u);
  CHECK(fixture.driver_state.converged[1] == 1u);
  CHECK(fixture.driver_state.iterations[1] == 1u);
  CHECK(std::abs(fixture.wavefunction.qsh[1] - 1.0) < 1.0e-14);
  CHECK(std::abs(fixture.wavefunction.qat[1] - 1.0) < 1.0e-14);
  return 0;
}

int test_structural_failure_atomicity_and_zero_allocation() {
  Fixture fixture;
  std::string error;
  error.reserve(256u);
  CHECK(make_fixture(1, fixture, error));
  const double before_qsh = fixture.wavefunction.qsh[0];
  const gpuxtb_status_t before_status = fixture.driver_state.system_statuses[0];
  SccDriverGeometryView malformed = fixture.geometry;
  malformed.h0_elements = 0;
  CHECK(iterate_scc_driver_batch_cpu(
            fixture.driver_plan, malformed, backend(), fixture.overlap_cache, fixture.wavefunction,
            fixture.mixer_state, fixture.driver_state, fixture.driver_scratch,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.wavefunction.qsh[0] == before_qsh);
  CHECK(fixture.driver_state.system_statuses[0] == before_status);
  CHECK(fixture.driver_state.iterations[0] == 0u);

  allocation_test::count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const gpuxtb_status_t status = iterate_scc_driver_batch_cpu(
      fixture.driver_plan, fixture.geometry, backend(), fixture.overlap_cache, fixture.wavefunction,
      fixture.mixer_state, fixture.driver_state, fixture.driver_scratch, error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  CHECK(status == GPUXTB_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == 0u);
  CHECK(fixture.driver_state.converged[0] == 1u);

  /* A correctly bound driver descriptor may still overlap a separately bound
   * mixer descriptor. Restart must reject that relationship before either
   * workspace or the wavefunction is modified. */
  SccDriverState overlapping_state;
  CHECK(bind_scc_driver_state(fixture.driver_plan, fixture.mixer_state_storage->data,
                              fixture.mixer_state_storage->size, overlapping_state,
                              error) == GPUXTB_STATUS_SUCCESS);
  std::vector<unsigned char> before_overlap(fixture.mixer_state_storage->size);
  std::memcpy(before_overlap.data(), fixture.mixer_state_storage->data,
              fixture.mixer_state_storage->size);
  const double qsh_before_restart = fixture.wavefunction.qsh[0];
  CHECK(restart_scc_driver_system_cpu(fixture.driver_plan, 0, fixture.wavefunction,
                                      fixture.mixer_state, overlapping_state,
                                      error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::memcmp(before_overlap.data(), fixture.mixer_state_storage->data,
                    fixture.mixer_state_storage->size) == 0);
  CHECK(fixture.wavefunction.qsh[0] == qsh_before_restart);
  return 0;
}

int test_unrestricted_is_rejected_without_spin_polarization() {
  const std::array<std::int64_t, 2> atom_offsets{{0, 1}};
  const std::array<std::int32_t, 1> atomic_numbers{{1}};
  const std::array<double, 1> charges{{1.0}};
  const std::array<std::int32_t, 1> unpaired{{0}};
  const std::array<std::int32_t, 1> spins{{2}};
  BasisPlan basis;
  IntegralPlan integrals;
  WavefunctionLayout wavefunction;
  ES2Plan es2;
  ES3Plan es3;
  AES2Plan aes2;
  MullikenPlan mulliken;
  EigensolverPlan eigensolver;
  SccMixerPlan mixer;
  SccDriverPlan driver;
  std::string error;
  CHECK(make_basis_plan(1, 1, atom_offsets.data(), atomic_numbers.data(), basis, error) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(make_integral_plan(basis, integrals, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(make_wavefunction_layout(basis, atomic_numbers.data(), charges.data(), unpaired.data(),
                                 spins.data(), wavefunction, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(make_es2_plan(basis, atomic_numbers.data(), es2, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(make_es3_plan(basis, atomic_numbers.data(), es3, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(make_aes2_plan(basis, atomic_numbers.data(), aes2, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(make_mulliken_plan(basis, integrals, wavefunction, mulliken, error) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(make_eigensolver_plan(wavefunction, eigensolver, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(make_scc_mixer_plan(wavefunction, 3, 0.4, 1.0e-10, 1.0e-10, mixer, error) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(make_scc_driver_plan(wavefunction, mulliken, es2, es3, aes2, eigensolver, mixer, 5u, 0.0,
                             driver, error) == GPUXTB_STATUS_NOT_SUPPORTED);
  CHECK(!driver.sealed());
  return 0;
}

int test_max_iteration_status_counts_attempts() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(1, fixture, error, 1u));
  fixture.wavefunction.qsh[0] = 0.0;
  fixture.wavefunction.qat[0] = 0.0;
  CHECK(restart_scc_driver_system_cpu(fixture.driver_plan, 0, fixture.wavefunction,
                                      fixture.mixer_state, fixture.driver_state,
                                      error) == GPUXTB_STATUS_SUCCESS);
  CHECK(iterate_scc_driver_batch_cpu(
            fixture.driver_plan, fixture.geometry, backend(), fixture.overlap_cache,
            fixture.wavefunction, fixture.mixer_state, fixture.driver_state, fixture.driver_scratch,
            error) == GPUXTB_STATUS_SCC_NOT_CONVERGED);
  CHECK(fixture.driver_state.iterations[0] == 1u);
  CHECK(fixture.mixer_state.iterations[0] == 1u);
  CHECK(fixture.driver_state.converged[0] == 0u);
  CHECK(fixture.driver_state.system_statuses[0] == GPUXTB_STATUS_SCC_NOT_CONVERGED);
  return 0;
}

int test_mixer_failure_preserves_public_history_and_counts_attempt() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(1, fixture, error));
  const double qsh_before = fixture.wavefunction.qsh[0];
  const double current_input_before = fixture.mixer_state.current_inputs[0];
  const double previous_input_before = fixture.mixer_state.previous_inputs[0];
  fixture.mixer_state.iterations[0] = std::numeric_limits<std::uint64_t>::max();
  CHECK(iterate_scc_driver_batch_cpu(
            fixture.driver_plan, fixture.geometry, backend(), fixture.overlap_cache,
            fixture.wavefunction, fixture.mixer_state, fixture.driver_state, fixture.driver_scratch,
            error) == GPUXTB_STATUS_INTERNAL_ERROR);
  CHECK(fixture.driver_state.iterations[0] == 1u);
  CHECK(fixture.driver_state.system_statuses[0] == GPUXTB_STATUS_INTERNAL_ERROR);
  CHECK(fixture.mixer_state.iterations[0] == std::numeric_limits<std::uint64_t>::max());
  CHECK(fixture.mixer_state.current_inputs[0] == current_input_before);
  CHECK(fixture.mixer_state.previous_inputs[0] == previous_input_before);
  CHECK(fixture.wavefunction.qsh[0] == qsh_before);
  CHECK(std::isnan(fixture.driver_state.free_energies[0]));
  return 0;
}

int test_converged_wavefunction_publishes_raw_mulliken_multipoles() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(1, fixture, error, 5u, 2.0));
  fixture.wavefunction.qsh[0] = 0.0;
  fixture.wavefunction.qat[0] = 0.0;
  CHECK(restart_scc_driver_system_cpu(fixture.driver_plan, 0, fixture.wavefunction,
                                      fixture.mixer_state, fixture.driver_state,
                                      error) == GPUXTB_STATUS_SUCCESS);
  CHECK(iterate_scc_driver_batch_cpu(fixture.driver_plan, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state,
                                     fixture.driver_scratch, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(fixture.driver_state.converged[0] == 1u);
  CHECK(std::abs(fixture.mixer_state.current_inputs[0] - 0.4) < 1.0e-14);
  CHECK(std::abs(fixture.wavefunction.qsh[0] - 1.0) < 1.0e-14);
  CHECK(std::abs(fixture.wavefunction.qat[0] - 1.0) < 1.0e-14);
  return 0;
}

int test_cached_explicit_point_charge_potential_enters_hamiltonian() {
  Fixture reference;
  Fixture embedded;
  std::string error;
  CHECK(make_fixture(1, reference, error));
  CHECK(make_fixture(1, embedded, error));
  const std::array<double, 1> point_charge_shell_potential{{0.25}};
  embedded.geometry.explicit_point_charge_shell_potential = point_charge_shell_potential.data();
  embedded.geometry.explicit_point_charge_shell_elements = 1;

  CHECK(iterate_scc_driver_batch_cpu(reference.driver_plan, reference.geometry, backend(),
                                     reference.overlap_cache, reference.wavefunction,
                                     reference.mixer_state, reference.driver_state,
                                     reference.driver_scratch, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(iterate_scc_driver_batch_cpu(embedded.driver_plan, embedded.geometry, backend(),
                                     embedded.overlap_cache, embedded.wavefunction,
                                     embedded.mixer_state, embedded.driver_state,
                                     embedded.driver_scratch, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(std::abs((embedded.wavefunction.eigenvalues[0] - reference.wavefunction.eigenvalues[0]) +
                 point_charge_shell_potential[0]) < 1.0e-14);
  return 0;
}

int test_component_chemistry_and_layout_mismatches_are_rejected() {
  std::string error;
  ComponentPlans ch;
  ComponentPlans hc;
  CHECK(make_component_plans({6, 1}, 1.0, 0, ch, error));
  CHECK(make_component_plans({1, 6}, 1.0, 0, hc, error));
  CHECK(ch.wavefunction.total_atoms == hc.wavefunction.total_atoms);
  CHECK(ch.wavefunction.total_shells == hc.wavefunction.total_shells);
  CHECK(ch.wavefunction.total_orbitals == hc.wavefunction.total_orbitals);
  CHECK(ch.wavefunction.electron_counts == hc.wavefunction.electron_counts);

  SccDriverPlan sentinel;
  CHECK(make_scc_driver_plan(ch.wavefunction, ch.mulliken, ch.es2, ch.es3, ch.aes2, ch.eigensolver,
                             ch.mixer, 5u, 0.0, sentinel, error) == GPUXTB_STATUS_SUCCESS);
  const SccDriverPlanData* const sentinel_identity = sentinel.identity();
  SccDriverPlan output = sentinel;

  CHECK(make_scc_driver_plan(ch.wavefunction, hc.mulliken, ch.es2, ch.es3, ch.aes2, ch.eigensolver,
                             ch.mixer, 5u, 0.0, output, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(output.identity() == sentinel_identity);
  CHECK(make_scc_driver_plan(ch.wavefunction, ch.mulliken, hc.es2, ch.es3, ch.aes2, ch.eigensolver,
                             ch.mixer, 5u, 0.0, output, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(output.identity() == sentinel_identity);
  CHECK(make_scc_driver_plan(ch.wavefunction, ch.mulliken, ch.es2, hc.es3, ch.aes2, ch.eigensolver,
                             ch.mixer, 5u, 0.0, output, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(output.identity() == sentinel_identity);
  CHECK(make_scc_driver_plan(ch.wavefunction, ch.mulliken, ch.es2, ch.es3, hc.aes2, ch.eigensolver,
                             ch.mixer, 5u, 0.0, output, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(output.identity() == sentinel_identity);

  ES3Plan modified_es3 = ch.es3;
  modified_es3.shell_gamma3[0] += 1.0;
  CHECK(make_scc_driver_plan(ch.wavefunction, ch.mulliken, ch.es2, modified_es3, ch.aes2,
                             ch.eigensolver, ch.mixer, 5u, 0.0, output,
                             error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(output.identity() == sentinel_identity);

  ComponentPlans h_plus;
  ComponentPlans neutral_h;
  CHECK(make_component_plans({1}, 1.0, 0, h_plus, error));
  CHECK(make_component_plans({1}, 0.0, 1, neutral_h, error));
  CHECK(make_scc_driver_plan(h_plus.wavefunction, h_plus.mulliken, h_plus.es2, h_plus.es3,
                             h_plus.aes2, neutral_h.eigensolver, h_plus.mixer, 5u, 0.0, output,
                             error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(output.identity() == sentinel_identity);

  ComponentPlans five_neon;
  ComponentPlans six_hydrogen;
  CHECK(make_component_plans({10, 10, 10, 10, 10}, 0.0, 0, five_neon, error));
  CHECK(make_component_plans({1, 1, 1, 1, 1, 1}, 0.0, 0, six_hydrogen, error));
  CHECK(five_neon.mixer.total_vector_elements() == six_hydrogen.mixer.total_vector_elements());
  SccDriverPlan neon_driver;
  CHECK(make_scc_driver_plan(five_neon.wavefunction, five_neon.mulliken, five_neon.es2,
                             five_neon.es3, five_neon.aes2, five_neon.eigensolver, five_neon.mixer,
                             5u, 0.0, neon_driver, error) == GPUXTB_STATUS_SUCCESS);
  output = neon_driver;
  const SccDriverPlanData* const neon_identity = neon_driver.identity();
  CHECK(make_scc_driver_plan(five_neon.wavefunction, five_neon.mulliken, five_neon.es2,
                             five_neon.es3, five_neon.aes2, five_neon.eigensolver,
                             six_hydrogen.mixer, 5u, 0.0, output,
                             error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(output.identity() == neon_identity);
  return 0;
}

int test_control_descriptors_cannot_alias_numerical_storage() {
  std::string error;
  {
    Fixture fixture;
    CHECK(make_fixture(1, fixture, error));
    auto* aliased_state =
        ::new (fixture.wavefunction_storage->data) SccDriverState(fixture.driver_state);
    std::vector<unsigned char> mixer_before(fixture.mixer_state_storage->size);
    std::vector<unsigned char> state_before(fixture.driver_state_storage->size);
    std::memcpy(mixer_before.data(), fixture.mixer_state_storage->data, mixer_before.size());
    std::memcpy(state_before.data(), fixture.driver_state_storage->data, state_before.size());
    CHECK(initialize_scc_driver_state_cpu(fixture.driver_plan, fixture.wavefunction,
                                          fixture.mixer_state, *aliased_state,
                                          error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(std::memcmp(mixer_before.data(), fixture.mixer_state_storage->data,
                      mixer_before.size()) == 0);
    CHECK(std::memcmp(state_before.data(), fixture.driver_state_storage->data,
                      state_before.size()) == 0);
  }
  {
    Fixture fixture;
    CHECK(make_fixture(1, fixture, error));
    auto* aliased_state =
        ::new (fixture.mixer_state_storage->data) SccDriverState(fixture.driver_state);
    std::vector<unsigned char> mixer_before(fixture.mixer_state_storage->size);
    std::vector<unsigned char> wavefunction_before(fixture.wavefunction_storage->size);
    std::memcpy(mixer_before.data(), fixture.mixer_state_storage->data, mixer_before.size());
    std::memcpy(wavefunction_before.data(), fixture.wavefunction_storage->data,
                wavefunction_before.size());
    CHECK(restart_scc_driver_system_cpu(fixture.driver_plan, 0, fixture.wavefunction,
                                        fixture.mixer_state, *aliased_state,
                                        error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(std::memcmp(mixer_before.data(), fixture.mixer_state_storage->data,
                      mixer_before.size()) == 0);
    CHECK(std::memcmp(wavefunction_before.data(), fixture.wavefunction_storage->data,
                      wavefunction_before.size()) == 0);
  }
  {
    Fixture fixture;
    CHECK(make_fixture(1, fixture, error));
    auto* aliased_wavefunction =
        ::new (fixture.driver_scratch_storage->data) WavefunctionView(fixture.wavefunction);
    std::vector<unsigned char> scratch_before(fixture.driver_scratch_storage->size);
    std::vector<unsigned char> state_before(fixture.driver_state_storage->size);
    std::memcpy(scratch_before.data(), fixture.driver_scratch_storage->data, scratch_before.size());
    std::memcpy(state_before.data(), fixture.driver_state_storage->data, state_before.size());
    CHECK(iterate_scc_driver_batch_cpu(
              fixture.driver_plan, fixture.geometry, backend(), fixture.overlap_cache,
              *aliased_wavefunction, fixture.mixer_state, fixture.driver_state,
              fixture.driver_scratch, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(std::memcmp(scratch_before.data(), fixture.driver_scratch_storage->data,
                      scratch_before.size()) == 0);
    CHECK(std::memcmp(state_before.data(), fixture.driver_state_storage->data,
                      state_before.size()) == 0);
  }
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_component_chemistry_and_layout_mismatches_are_rejected();
      status != 0) {
    return status;
  }
  if (const int status = test_control_descriptors_cannot_alias_numerical_storage(); status != 0) {
    return status;
  }
  if (const int status = test_unrestricted_is_rejected_without_spin_polarization(); status != 0) {
    return status;
  }
  if (const int status = test_max_iteration_status_counts_attempts(); status != 0) {
    return status;
  }
  if (const int status = test_mixer_failure_preserves_public_history_and_counts_attempt();
      status != 0) {
    return status;
  }
  if (const int status = test_converged_wavefunction_publishes_raw_mulliken_multipoles();
      status != 0) {
    return status;
  }
  if (const int status = test_cached_explicit_point_charge_potential_enters_hamiltonian();
      status != 0) {
    return status;
  }
  if (const int status = test_ragged_failure_isolation_restart_and_skip(); status != 0) {
    return status;
  }
  return test_structural_failure_atomicity_and_zero_allocation();
}
