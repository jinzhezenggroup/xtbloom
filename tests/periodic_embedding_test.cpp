#include "model/gfn2/periodic_embedding.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace allocation_test {
std::atomic<std::size_t> count{0};
std::atomic<bool> enabled{false};
}  // namespace allocation_test

#if defined(__GNUC__) || defined(__clang__)
#define XTBLOOM_TEST_NOINLINE __attribute__((noinline))
#else
#define XTBLOOM_TEST_NOINLINE
#endif

XTBLOOM_TEST_NOINLINE
void* operator new(std::size_t size) {
  if (allocation_test::enabled.load(std::memory_order_relaxed)) {
    allocation_test::count.fetch_add(1u, std::memory_order_relaxed);
  }
  if (void* pointer = std::malloc(size == 0u ? 1u : size); pointer != nullptr) {
    return pointer;
  }
  throw std::bad_alloc();
}

XTBLOOM_TEST_NOINLINE void* operator new[](std::size_t size) { return ::operator new(size); }
XTBLOOM_TEST_NOINLINE void operator delete(void* pointer) noexcept { std::free(pointer); }
XTBLOOM_TEST_NOINLINE void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }
XTBLOOM_TEST_NOINLINE void operator delete(void* pointer, std::size_t) noexcept {
  ::operator delete(pointer);
}
XTBLOOM_TEST_NOINLINE void operator delete[](void* pointer, std::size_t) noexcept {
  ::operator delete[](pointer);
}

#undef XTBLOOM_TEST_NOINLINE

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

using xtbloom::detail::gfn2::PeriodicEmbeddingPlan;
using xtbloom::detail::gfn2::PeriodicEmbeddingView;
using xtbloom::detail::gfn2::PeriodicEmbeddingWorkspace;

static_assert(std::is_trivially_copyable_v<PeriodicEmbeddingView>);
static_assert(std::is_standard_layout_v<PeriodicEmbeddingView>);
static_assert(std::is_trivially_copyable_v<PeriodicEmbeddingWorkspace>);
static_assert(std::is_standard_layout_v<PeriodicEmbeddingWorkspace>);
static_assert(std::is_nothrow_copy_constructible_v<PeriodicEmbeddingPlan>);
static_assert(sizeof(PeriodicEmbeddingPlan) <= 4u * sizeof(void*));

struct Fixture {
  PeriodicEmbeddingPlan plan;
  std::vector<double> shifts;
  std::vector<double> matrices;
  std::vector<double> charges;
  std::vector<double> potentials;
  std::vector<double> energies;
  std::vector<xtbloom_status_t> statuses;
  std::vector<double> scratch;
  PeriodicEmbeddingView view;
  PeriodicEmbeddingWorkspace workspace;
};

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

bool same_view(const PeriodicEmbeddingView& actual, const PeriodicEmbeddingView& expected) {
  return actual.shifts == expected.shifts && actual.shift_elements == expected.shift_elements &&
         actual.response_matrices == expected.response_matrices &&
         actual.response_elements == expected.response_elements &&
         actual.atomic_charges == expected.atomic_charges &&
         actual.charge_elements == expected.charge_elements &&
         actual.atomic_potentials == expected.atomic_potentials &&
         actual.potential_elements == expected.potential_elements &&
         actual.energies == expected.energies &&
         actual.energy_elements == expected.energy_elements &&
         actual.system_statuses == expected.system_statuses &&
         actual.status_elements == expected.status_elements &&
         actual.plan_identity == expected.plan_identity;
}

bool same_workspace(const PeriodicEmbeddingWorkspace& actual,
                    const PeriodicEmbeddingWorkspace& expected) {
  return actual.potential_scratch == expected.potential_scratch &&
         actual.potential_elements == expected.potential_elements &&
         actual.plan_identity == expected.plan_identity;
}

std::size_t matrix_elements(const std::vector<std::int64_t>& offsets) {
  std::size_t result = 0u;
  for (std::size_t system = 0u; system + 1u < offsets.size(); ++system) {
    const auto atoms = static_cast<std::size_t>(offsets[system + 1u] - offsets[system]);
    result += atoms * atoms;
  }
  return result;
}

bool make_fixture(const std::vector<std::int64_t>& offsets, std::vector<double> shifts,
                  std::vector<double> matrices, std::vector<double> charges, Fixture& fixture,
                  std::string& error) {
  const std::int64_t batch_size = static_cast<std::int64_t>(offsets.size() - 1u);
  const std::int64_t atom_count = offsets.back();
  if (shifts.size() != static_cast<std::size_t>(atom_count) ||
      charges.size() != static_cast<std::size_t>(atom_count) ||
      matrices.size() != matrix_elements(offsets) ||
      xtbloom::detail::gfn2::make_periodic_embedding_plan(
          batch_size, atom_count, offsets.data(), fixture.plan, error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  fixture.shifts = std::move(shifts);
  fixture.matrices = std::move(matrices);
  fixture.charges = std::move(charges);
  fixture.potentials.assign(static_cast<std::size_t>(atom_count), 71.0);
  fixture.energies.assign(static_cast<std::size_t>(batch_size), 73.0);
  fixture.statuses.assign(static_cast<std::size_t>(batch_size), static_cast<xtbloom_status_t>(79));
  fixture.scratch.resize(static_cast<std::size_t>(fixture.plan.maximum_atoms()));
  if (xtbloom::detail::gfn2::bind_periodic_embedding_view(
          fixture.plan, fixture.shifts.data(), fixture.shifts.size(), fixture.matrices.data(),
          fixture.matrices.size(), fixture.charges.data(), fixture.charges.size(),
          fixture.potentials.data(), fixture.potentials.size(), fixture.energies.data(),
          fixture.energies.size(), fixture.statuses.data(), fixture.statuses.size(), fixture.view,
          error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::bind_periodic_embedding_workspace(
          fixture.plan, fixture.scratch.data(), fixture.scratch.size(), fixture.workspace, error) !=
          XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  return true;
}

void reference_system(const Fixture& fixture, std::size_t system, std::vector<double>& potentials,
                      double& energy) {
  const std::int64_t begin = fixture.plan.atom_offsets()[system];
  const std::int64_t end = fixture.plan.atom_offsets()[system + 1u];
  const std::int64_t atoms = end - begin;
  const std::int64_t matrix_begin = fixture.plan.matrix_offsets()[system];
  energy = 0.0;
  for (std::int64_t row = 0; row < atoms; ++row) {
    double response = 0.0;
    for (std::int64_t column = 0; column < atoms; ++column) {
      response =
          std::fma(fixture.matrices[static_cast<std::size_t>(matrix_begin + row * atoms + column)],
                   fixture.charges[static_cast<std::size_t>(begin + column)], response);
    }
    const std::size_t atom = static_cast<std::size_t>(begin + row);
    potentials[atom] = fixture.shifts[atom] + response;
    energy = std::fma(fixture.charges[atom], fixture.shifts[atom], energy);
    energy = std::fma(0.5 * fixture.charges[atom], response, energy);
  }
}

int test_ragged_oracle_and_empty_member() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 2, 2, 5}, {0.1, -0.2, 0.3, -0.1, 0.05},
                     {2.0, 0.25, 0.25, 1.0, 1.0, 0.1, -0.2, 0.1, 1.5, 0.3, -0.2, 0.3, 0.8},
                     {0.4, -0.5, 0.2, 0.3, -0.1}, fixture, error));
  CHECK(fixture.plan.batch_size() == 3);
  CHECK(fixture.plan.total_atoms() == 5);
  CHECK(fixture.plan.total_matrix_elements() == 13);
  CHECK(fixture.plan.maximum_atoms() == 3);
  CHECK(fixture.plan.atom_offsets() == std::vector<std::int64_t>({0, 2, 2, 5}));
  CHECK(fixture.plan.matrix_offsets() == std::vector<std::int64_t>({0, 4, 4, 13}));

  CHECK(xtbloom::detail::gfn2::evaluate_periodic_embedding_batch_cpu(
            fixture.plan, fixture.view, fixture.workspace, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(error.empty());
  std::vector<double> expected_potentials(5u);
  std::vector<double> expected_energies(3u);
  for (std::size_t system = 0u; system < expected_energies.size(); ++system) {
    reference_system(fixture, system, expected_potentials, expected_energies[system]);
  }
  CHECK(fixture.potentials == expected_potentials);
  for (std::size_t system = 0u; system < expected_energies.size(); ++system) {
    CHECK(near(fixture.energies[system], expected_energies[system], 2.0e-16));
  }
  CHECK(fixture.energies[1] == 0.0);
  CHECK(std::all_of(fixture.statuses.begin(), fixture.statuses.end(),
                    [](xtbloom_status_t status) { return status == XTBLOOM_STATUS_SUCCESS; }));

  const PeriodicEmbeddingPlan copied = fixture.plan;
  CHECK(copied.identity() == fixture.plan.identity());
  CHECK(copied.atom_offsets() == fixture.plan.atom_offsets());
  return 0;
}

double evaluate_energy(Fixture& fixture, std::string& error) {
  if (xtbloom::detail::gfn2::evaluate_periodic_embedding_batch_cpu(
          fixture.plan, fixture.view, fixture.workspace, error) != XTBLOOM_STATUS_SUCCESS) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return fixture.energies[0];
}

int test_q_b_and_symmetric_a_finite_differences() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 3}, {0.14, -0.23, 0.07},
                     {1.1, -0.2, 0.35, -0.2, 0.8, 0.12, 0.35, 0.12, 1.4}, {0.31, -0.27, 0.16},
                     fixture, error));
  CHECK(evaluate_energy(fixture, error) == fixture.energies[0]);
  const std::vector<double> potential = fixture.potentials;
  constexpr double step = 1.0e-6;

  for (std::size_t atom = 0u; atom < fixture.charges.size(); ++atom) {
    const double original = fixture.charges[atom];
    fixture.charges[atom] = original + step;
    const double plus = evaluate_energy(fixture, error);
    fixture.charges[atom] = original - step;
    const double minus = evaluate_energy(fixture, error);
    fixture.charges[atom] = original;
    CHECK(near((plus - minus) / (2.0 * step), potential[atom], 4.0e-11));
  }

  for (std::size_t atom = 0u; atom < fixture.shifts.size(); ++atom) {
    const double original = fixture.shifts[atom];
    fixture.shifts[atom] = original + step;
    const double plus = evaluate_energy(fixture, error);
    fixture.shifts[atom] = original - step;
    const double minus = evaluate_energy(fixture, error);
    fixture.shifts[atom] = original;
    CHECK(near((plus - minus) / (2.0 * step), fixture.charges[atom], 3.0e-11));
  }

  for (std::size_t row = 0u; row < 3u; ++row) {
    for (std::size_t column = row; column < 3u; ++column) {
      const std::size_t first = row * 3u + column;
      const std::size_t second = column * 3u + row;
      const double original = fixture.matrices[first];
      fixture.matrices[first] = original + step;
      fixture.matrices[second] = original + step;
      const double plus = evaluate_energy(fixture, error);
      fixture.matrices[first] = original - step;
      fixture.matrices[second] = original - step;
      const double minus = evaluate_energy(fixture, error);
      fixture.matrices[first] = original;
      fixture.matrices[second] = original;
      const double expected = row == column ? 0.5 * fixture.charges[row] * fixture.charges[row]
                                            : fixture.charges[row] * fixture.charges[column];
      CHECK(near((plus - minus) / (2.0 * step), expected, 3.0e-11));
    }
  }
  return 0;
}

int test_batch_equals_sequential_and_system_worker() {
  Fixture batch;
  std::string error;
  CHECK(make_fixture({0, 1, 3, 3}, {0.2, -0.1, 0.3}, {1.7, 0.8, -0.15, -0.15, 1.2},
                     {0.4, -0.25, 0.35}, batch, error));
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_embedding_batch_cpu(
            batch.plan, batch.view, batch.workspace, error) == XTBLOOM_STATUS_SUCCESS);

  for (std::size_t system = 0u; system < 3u; ++system) {
    const std::int64_t atom_begin = batch.plan.atom_offsets()[system];
    const std::int64_t atom_end = batch.plan.atom_offsets()[system + 1u];
    const std::int64_t matrix_begin = batch.plan.matrix_offsets()[system];
    const std::int64_t matrix_end = batch.plan.matrix_offsets()[system + 1u];
    const std::vector<std::int64_t> offsets{0, atom_end - atom_begin};
    Fixture sequential;
    CHECK(make_fixture(
        offsets,
        std::vector<double>(batch.shifts.begin() + atom_begin, batch.shifts.begin() + atom_end),
        std::vector<double>(batch.matrices.begin() + matrix_begin,
                            batch.matrices.begin() + matrix_end),
        std::vector<double>(batch.charges.begin() + atom_begin, batch.charges.begin() + atom_end),
        sequential, error));
    CHECK(xtbloom::detail::gfn2::evaluate_periodic_embedding_system_cpu(
              sequential.plan, 0, sequential.view, sequential.workspace, error) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(std::equal(sequential.potentials.begin(), sequential.potentials.end(),
                     batch.potentials.begin() + atom_begin));
    CHECK(sequential.energies[0] == batch.energies[system]);
    CHECK(sequential.statuses[0] == batch.statuses[system]);
  }
  return 0;
}

int test_parallel_workers_match_serial_exactly() {
  const std::vector<std::int64_t> offsets{0, 3, 5};
  const std::vector<double> shifts{0.12, -0.08, 0.21, -0.17, 0.04};
  const std::vector<double> matrices{
      1.0, 0.1, -0.2, 0.1, 1.3, 0.25, -0.2, 0.25, 0.9, 0.75, 0.3, 0.3, 1.1,
  };
  const std::vector<double> charges{0.3, -0.2, 0.4, -0.35, 0.15};
  Fixture serial;
  Fixture parallel;
  std::string error;
  CHECK(make_fixture(offsets, shifts, matrices, charges, serial, error));
  CHECK(make_fixture(offsets, shifts, matrices, charges, parallel, error));
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_embedding_batch_cpu(
            serial.plan, serial.view, serial.workspace, error) == XTBLOOM_STATUS_SUCCESS);

  std::vector<double> first_scratch(static_cast<std::size_t>(parallel.plan.maximum_atoms()));
  std::vector<double> second_scratch(static_cast<std::size_t>(parallel.plan.maximum_atoms()));
  PeriodicEmbeddingWorkspace first_workspace;
  PeriodicEmbeddingWorkspace second_workspace;
  std::string first_error;
  std::string second_error;
  CHECK(xtbloom::detail::gfn2::bind_periodic_embedding_workspace(
            parallel.plan, first_scratch.data(), first_scratch.size(), first_workspace,
            first_error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::bind_periodic_embedding_workspace(
            parallel.plan, second_scratch.data(), second_scratch.size(), second_workspace,
            second_error) == XTBLOOM_STATUS_SUCCESS);

  std::atomic<unsigned int> ready{0u};
  std::atomic<bool> go{false};
  xtbloom_status_t first_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  xtbloom_status_t second_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  const auto await_start = [&]() {
    ready.fetch_add(1u, std::memory_order_release);
    while (!go.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  };
  std::thread first([&]() {
    await_start();
    first_status = xtbloom::detail::gfn2::evaluate_periodic_embedding_system_cpu(
        parallel.plan, 0, parallel.view, first_workspace, first_error);
  });
  std::thread second([&]() {
    await_start();
    second_status = xtbloom::detail::gfn2::evaluate_periodic_embedding_system_cpu(
        parallel.plan, 1, parallel.view, second_workspace, second_error);
  });
  while (ready.load(std::memory_order_acquire) != 2u) {
    std::this_thread::yield();
  }
  go.store(true, std::memory_order_release);
  first.join();
  second.join();

  CHECK(first_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(second_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(first_error.empty() && second_error.empty());
  CHECK(parallel.potentials == serial.potentials);
  CHECK(parallel.energies == serial.energies);
  CHECK(parallel.statuses == serial.statuses);
  return 0;
}

int test_all_empty_batch() {
  PeriodicEmbeddingPlan plan;
  PeriodicEmbeddingView view;
  PeriodicEmbeddingWorkspace workspace;
  std::vector<double> energies(2u, 9.0);
  std::vector<xtbloom_status_t> statuses(2u, static_cast<xtbloom_status_t>(17));
  const std::int64_t offsets[] = {0, 0, 0};
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_periodic_embedding_plan(2, 0, offsets, plan, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::bind_periodic_embedding_view(
            plan, nullptr, 0u, nullptr, 0u, nullptr, 0u, nullptr, 0u, energies.data(),
            energies.size(), statuses.data(), statuses.size(), view,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::bind_periodic_embedding_workspace(plan, nullptr, 0u, workspace,
                                                                 error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_embedding_batch_cpu(
            plan, view, workspace, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(energies == std::vector<double>({0.0, 0.0}));
  CHECK(statuses ==
        std::vector<xtbloom_status_t>({XTBLOOM_STATUS_SUCCESS, XTBLOOM_STATUS_SUCCESS}));
  return 0;
}

int test_structural_atomicity_alias_and_overflow_rejection() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 2}, {0.2, -0.4}, {1.0, 0.3, 0.3, 0.9}, {0.5, -0.2}, fixture, error));
  const std::vector<double> potential_before = fixture.potentials;
  const std::vector<double> energy_before = fixture.energies;
  const std::vector<xtbloom_status_t> status_before = fixture.statuses;

  PeriodicEmbeddingView malformed = fixture.view;
  malformed.potential_elements = 1;
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_embedding_batch_cpu(
            fixture.plan, malformed, fixture.workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.potentials == potential_before && fixture.energies == energy_before &&
        fixture.statuses == status_before);

  malformed = fixture.view;
  malformed.atomic_potentials = const_cast<double*>(malformed.atomic_charges);
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_embedding_batch_cpu(
            fixture.plan, malformed, fixture.workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.potentials == potential_before && fixture.energies == energy_before &&
        fixture.statuses == status_before);

  PeriodicEmbeddingWorkspace alias_workspace = fixture.workspace;
  alias_workspace.potential_scratch = fixture.potentials.data();
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_embedding_batch_cpu(
            fixture.plan, fixture.view, alias_workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.potentials == potential_before && fixture.energies == energy_before &&
        fixture.statuses == status_before);

  PeriodicEmbeddingView unchanged = fixture.view;
  CHECK(xtbloom::detail::gfn2::bind_periodic_embedding_view(
            fixture.plan, fixture.shifts.data(), 1u, fixture.matrices.data(),
            fixture.matrices.size(), fixture.charges.data(), fixture.charges.size(),
            fixture.potentials.data(), fixture.potentials.size(), fixture.energies.data(),
            fixture.energies.size(), fixture.statuses.data(), fixture.statuses.size(), unchanged,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(same_view(unchanged, fixture.view));

  PeriodicEmbeddingWorkspace unchanged_workspace = fixture.workspace;
  CHECK(xtbloom::detail::gfn2::bind_periodic_embedding_workspace(
            fixture.plan, nullptr, fixture.scratch.size(), unchanged_workspace, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(same_workspace(unchanged_workspace, fixture.workspace));

  const auto overflowing_address =
      std::numeric_limits<std::uintptr_t>::max() - (alignof(double) - 1u);
  const auto* const overflowing_pointer = reinterpret_cast<const double*>(overflowing_address);
  unchanged = fixture.view;
  CHECK(xtbloom::detail::gfn2::bind_periodic_embedding_view(
            fixture.plan, overflowing_pointer, fixture.shifts.size(), fixture.matrices.data(),
            fixture.matrices.size(), fixture.charges.data(), fixture.charges.size(),
            fixture.potentials.data(), fixture.potentials.size(), fixture.energies.data(),
            fixture.energies.size(), fixture.statuses.data(), fixture.statuses.size(), unchanged,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(same_view(unchanged, fixture.view));
  CHECK(xtbloom::detail::gfn2::bind_periodic_embedding_view(
            fixture.plan, fixture.shifts.data(), fixture.shifts.size(), fixture.matrices.data(),
            fixture.matrices.size(), fixture.charges.data(), fixture.charges.size(),
            fixture.charges.data(), fixture.potentials.size(), fixture.energies.data(),
            fixture.energies.size(), fixture.statuses.data(), fixture.statuses.size(), unchanged,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(same_view(unchanged, fixture.view));

  const auto identity = fixture.plan.identity();
  const std::int64_t invalid_offsets[] = {0, 3};
  CHECK(xtbloom::detail::gfn2::make_periodic_embedding_plan(
            1, 2, invalid_offsets, fixture.plan, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.plan.identity() == identity);
  const std::int64_t nonmonotone[] = {0, 2, 1};
  CHECK(xtbloom::detail::gfn2::make_periodic_embedding_plan(
            2, 1, nonmonotone, fixture.plan, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.plan.identity() == identity);
  const std::int64_t overflowing[] = {0, 3037000500LL};
  CHECK(xtbloom::detail::gfn2::make_periodic_embedding_plan(
            1, 3037000500LL, overflowing, fixture.plan, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.plan.identity() == identity);

  alignas(std::int64_t) unsigned char offset_storage[3u * sizeof(std::int64_t)]{};
  CHECK(xtbloom::detail::gfn2::make_periodic_embedding_plan(
            1, 0, reinterpret_cast<const std::int64_t*>(offset_storage + 1u), fixture.plan,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.plan.identity() == identity);
  return 0;
}

int test_per_system_numerical_isolation() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 2, 3, 5}, {0.1, -0.2, std::numeric_limits<double>::quiet_NaN(), 0.3, -0.4},
                     {1.0, 0.2, 0.2, 0.8, 1.1, std::numeric_limits<double>::max(), 0.0, 0.0, 1.0},
                     {0.4, -0.1, 0.2, 2.0, 0.5}, fixture, error));
  const std::vector<double> potential_before = fixture.potentials;
  const std::vector<double> energy_before = fixture.energies;
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_embedding_batch_cpu(
            fixture.plan, fixture.view, fixture.workspace, error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(fixture.statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.statuses[1] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(fixture.statuses[2] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(fixture.potentials[0] != potential_before[0] &&
        fixture.potentials[1] != potential_before[1]);
  CHECK(fixture.energies[0] != energy_before[0]);
  CHECK(fixture.potentials[2] == potential_before[2] && fixture.energies[1] == energy_before[1]);
  CHECK(fixture.potentials[3] == potential_before[3] &&
        fixture.potentials[4] == potential_before[4] && fixture.energies[2] == energy_before[2]);

  fixture.shifts[2] = 0.05;
  fixture.matrices[5] = 1.0;
  fixture.matrices[6] = 0.4;
  fixture.matrices[7] = -0.2;
  fixture.matrices[8] = 1.0;
  const double failed_potential_before = fixture.potentials[3];
  const double failed_energy_before = fixture.energies[2];
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_embedding_system_cpu(fixture.plan, 2, fixture.view,
                                                                      fixture.workspace, error) ==
        XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(fixture.statuses[2] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(fixture.potentials[3] == failed_potential_before &&
        fixture.energies[2] == failed_energy_before);
  return 0;
}

int test_one_ulp_asymmetry_is_rejected() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 2}, {0.1, -0.3}, {1.0, 0.0, -0.0, 0.8}, {0.4, -0.2}, fixture, error));
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_embedding_batch_cpu(
            fixture.plan, fixture.view, fixture.workspace, error) == XTBLOOM_STATUS_SUCCESS);
  const std::vector<double> potential_before = fixture.potentials;
  const std::vector<double> energy_before = fixture.energies;
  fixture.matrices[1] = 0.25;
  fixture.matrices[2] =
      std::nextafter(fixture.matrices[1], std::numeric_limits<double>::infinity());
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_embedding_batch_cpu(
            fixture.plan, fixture.view, fixture.workspace, error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(fixture.statuses[0] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(fixture.potentials == potential_before);
  CHECK(fixture.energies == energy_before);
  return 0;
}

int test_zero_allocation_hot_path() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 2, 2}, {0.1, -0.2}, {1.0, 0.15, 0.15, 0.7}, {0.3, -0.4}, fixture, error));
  CHECK(xtbloom::detail::gfn2::evaluate_periodic_embedding_batch_cpu(
            fixture.plan, fixture.view, fixture.workspace, error) == XTBLOOM_STATUS_SUCCESS);
  allocation_test::count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const xtbloom_status_t batch_status =
      xtbloom::detail::gfn2::evaluate_periodic_embedding_batch_cpu(fixture.plan, fixture.view,
                                                                   fixture.workspace, error);
  const xtbloom_status_t worker_status =
      xtbloom::detail::gfn2::evaluate_periodic_embedding_system_cpu(fixture.plan, 0, fixture.view,
                                                                    fixture.workspace, error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  CHECK(batch_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(worker_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == 0u);
  return 0;
}

}  // namespace

int main() {
  if (const int line = test_ragged_oracle_and_empty_member(); line != 0) {
    return line;
  }
  if (const int line = test_q_b_and_symmetric_a_finite_differences(); line != 0) {
    return line;
  }
  if (const int line = test_batch_equals_sequential_and_system_worker(); line != 0) {
    return line;
  }
  if (const int line = test_parallel_workers_match_serial_exactly(); line != 0) {
    return line;
  }
  if (const int line = test_all_empty_batch(); line != 0) {
    return line;
  }
  if (const int line = test_structural_atomicity_alias_and_overflow_rejection(); line != 0) {
    return line;
  }
  if (const int line = test_per_system_numerical_isolation(); line != 0) {
    return line;
  }
  if (const int line = test_one_ulp_asymmetry_is_rejected(); line != 0) {
    return line;
  }
  if (const int line = test_zero_allocation_hot_path(); line != 0) {
    return line;
  }
  return 0;
}
