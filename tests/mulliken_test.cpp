#include "model/gfn2/mulliken.hpp"

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
#include <type_traits>
#include <vector>

namespace allocation_test {
std::atomic<std::size_t> count{0};
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

#if defined(__GNUC__) && !defined(__clang__)
#define GPUXTB_TEST_NOINLINE __attribute__((noinline))
#else
#define GPUXTB_TEST_NOINLINE
#endif

/* GCC 11 diagnoses the intentional malloc/free implementation as a mismatched
 * pair only after inlining this test-only allocation counter shim. */
GPUXTB_TEST_NOINLINE void operator delete(void* pointer) noexcept { std::free(pointer); }

void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }

void operator delete(void* pointer, std::size_t) noexcept { ::operator delete(pointer); }

void operator delete[](void* pointer, std::size_t) noexcept { ::operator delete[](pointer); }

#undef GPUXTB_TEST_NOINLINE

#define CHECK(condition)                                                                   \
  do {                                                                                     \
    if (!(condition)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      return __LINE__;                                                                     \
    }                                                                                      \
  } while (false)

namespace {

using gpuxtb::detail::gfn2::BasisPlan;
using gpuxtb::detail::gfn2::IntegralPlan;
using gpuxtb::detail::gfn2::MullikenDensityView;
using gpuxtb::detail::gfn2::MullikenHamiltonianView;
using gpuxtb::detail::gfn2::MullikenIntegralView;
using gpuxtb::detail::gfn2::MullikenPlan;
using gpuxtb::detail::gfn2::MullikenPopulationView;
using gpuxtb::detail::gfn2::MullikenPotentialView;
using gpuxtb::detail::gfn2::MullikenWorkspace;
using gpuxtb::detail::gfn2::WavefunctionLayout;

static_assert(std::is_trivially_copyable_v<MullikenIntegralView>);
static_assert(std::is_trivially_copyable_v<MullikenDensityView>);
static_assert(std::is_trivially_copyable_v<MullikenPopulationView>);
static_assert(std::is_trivially_copyable_v<MullikenPotentialView>);
static_assert(std::is_trivially_copyable_v<MullikenHamiltonianView>);
static_assert(std::is_trivially_copyable_v<MullikenWorkspace>);

bool near(double actual, double expected, double tolerance = 2.0e-13) {
  return std::abs(actual - expected) <=
         tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

bool all_near(const std::vector<double>& actual, const std::vector<double>& expected,
              double tolerance = 2.0e-13) {
  return actual.size() == expected.size() &&
         std::equal(
             actual.begin(), actual.end(), expected.begin(),
             [tolerance](double left, double right) { return near(left, right, tolerance); });
}

bool all_equal_to(const std::vector<double>& values, double expected) {
  return std::all_of(values.begin(), values.end(),
                     [expected](double value) { return value == expected; });
}

struct Fixture {
  BasisPlan basis;
  IntegralPlan integral_plan;
  WavefunctionLayout wavefunction;
  MullikenPlan plan;
  std::vector<double> overlap;
  std::vector<double> dipole_integrals;
  std::vector<double> quadrupole_integrals;
  std::vector<double> density;
  std::vector<double> qsh;
  std::vector<double> qat;
  std::vector<double> dipole;
  std::vector<double> quadrupole;
  std::vector<double> vat;
  std::vector<double> vsh;
  std::vector<double> dipole_potential;
  std::vector<double> quadrupole_potential;
  std::vector<double> hamiltonian;
  std::vector<double> scratch;
};

bool make_fixture(const std::vector<std::int64_t>& atom_offsets,
                  const std::vector<std::int32_t>& atomic_numbers,
                  const std::vector<double>& charges, const std::vector<std::int32_t>& unpaired,
                  const std::vector<std::int32_t>& spin_channels, Fixture& fixture,
                  std::string& error) {
  const std::int64_t batch_size = static_cast<std::int64_t>(atom_offsets.size() - 1u);
  if (gpuxtb::detail::gfn2::make_basis_plan(
          batch_size, static_cast<std::int64_t>(atomic_numbers.size()), atom_offsets.data(),
          atomic_numbers.data(), fixture.basis, error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::make_integral_plan(fixture.basis, fixture.integral_plan, error) !=
          GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::make_wavefunction_layout(
          fixture.basis, atomic_numbers.data(), charges.data(), unpaired.data(),
          spin_channels.data(), fixture.wavefunction, error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::make_mulliken_plan(fixture.basis, fixture.integral_plan,
                                               fixture.wavefunction, fixture.plan,
                                               error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }

  fixture.overlap.resize(static_cast<std::size_t>(fixture.plan.matrix_elements()));
  fixture.dipole_integrals.resize(static_cast<std::size_t>(3 * fixture.plan.matrix_elements()));
  fixture.quadrupole_integrals.resize(static_cast<std::size_t>(6 * fixture.plan.matrix_elements()));
  fixture.density.resize(static_cast<std::size_t>(fixture.plan.density_elements()));
  fixture.qsh.resize(static_cast<std::size_t>(fixture.plan.shell_population_elements()));
  fixture.qat.resize(static_cast<std::size_t>(fixture.plan.atom_population_elements()));
  fixture.dipole.resize(static_cast<std::size_t>(fixture.plan.dipole_population_elements()));
  fixture.quadrupole.resize(
      static_cast<std::size_t>(fixture.plan.quadrupole_population_elements()));
  fixture.vat.resize(fixture.qat.size());
  fixture.vsh.resize(fixture.qsh.size());
  fixture.dipole_potential.resize(fixture.dipole.size());
  fixture.quadrupole_potential.resize(fixture.quadrupole.size());
  fixture.hamiltonian.resize(fixture.density.size());
  fixture.scratch.resize(static_cast<std::size_t>(std::max(
      fixture.plan.population_scratch_elements(), fixture.plan.hamiltonian_scratch_elements())));
  return true;
}

MullikenIntegralView integral_view(const Fixture& fixture) {
  return {fixture.overlap.data(), fixture.dipole_integrals.data(),
          fixture.quadrupole_integrals.data(), fixture.plan.matrix_elements(),
          fixture.plan.identity()};
}

MullikenDensityView density_view(const Fixture& fixture) {
  return {fixture.density.data(), fixture.plan.density_elements(), fixture.plan.identity()};
}

MullikenPopulationView population_view(Fixture& fixture) {
  return {fixture.qsh.data(),        fixture.plan.shell_population_elements(),
          fixture.qat.data(),        fixture.plan.atom_population_elements(),
          fixture.dipole.data(),     fixture.plan.dipole_population_elements(),
          fixture.quadrupole.data(), fixture.plan.quadrupole_population_elements(),
          fixture.plan.identity()};
}

MullikenPotentialView potential_view(const Fixture& fixture) {
  return {fixture.vat.data(),
          fixture.plan.atom_population_elements(),
          fixture.vsh.data(),
          fixture.plan.shell_population_elements(),
          fixture.dipole_potential.data(),
          fixture.plan.dipole_population_elements(),
          fixture.quadrupole_potential.data(),
          fixture.plan.quadrupole_population_elements(),
          fixture.plan.identity()};
}

MullikenHamiltonianView hamiltonian_view(Fixture& fixture) {
  return {fixture.hamiltonian.data(), fixture.plan.density_elements(), fixture.plan.identity()};
}

MullikenWorkspace workspace_view(Fixture& fixture) {
  return {fixture.scratch.data(), static_cast<std::int64_t>(fixture.scratch.size())};
}

bool evaluate_population(Fixture& fixture, std::string& error) {
  return gpuxtb::detail::gfn2::evaluate_mulliken_population_cpu(
             fixture.plan, integral_view(fixture), density_view(fixture), population_view(fixture),
             workspace_view(fixture), error) == GPUXTB_STATUS_SUCCESS;
}

bool assemble_hamiltonian(Fixture& fixture, std::string& error) {
  return gpuxtb::detail::gfn2::add_mulliken_hamiltonian_cpu(
             fixture.plan, integral_view(fixture), potential_view(fixture),
             hamiltonian_view(fixture), workspace_view(fixture), error) == GPUXTB_STATUS_SUCCESS;
}

template <typename T>
std::array<std::byte, sizeof(T)> object_bytes(const T& value) {
  std::array<std::byte, sizeof(T)> result{};
  std::memcpy(result.data(), &value, sizeof(T));
  return result;
}

template <typename T>
bool object_bytes_equal(const T& value, const std::array<std::byte, sizeof(T)>& expected) {
  return std::memcmp(&value, expected.data(), sizeof(T)) == 0;
}

gpuxtb_status_t attempt_population_control_alias(Fixture& fixture, double* candidate,
                                                 std::size_t buffer, std::string& error) {
  std::fill(fixture.qsh.begin(), fixture.qsh.end(), 71.0);
  std::fill(fixture.qat.begin(), fixture.qat.end(), 72.0);
  std::fill(fixture.dipole.begin(), fixture.dipole.end(), 73.0);
  std::fill(fixture.quadrupole.begin(), fixture.quadrupole.end(), 74.0);
  MullikenIntegralView integrals = integral_view(fixture);
  MullikenDensityView density = density_view(fixture);
  MullikenPopulationView population = population_view(fixture);
  MullikenWorkspace workspace = workspace_view(fixture);
  switch (buffer) {
    case 0:
      integrals.overlap = candidate;
      break;
    case 1:
      integrals.dipole = candidate;
      break;
    case 2:
      integrals.quadrupole = candidate;
      break;
    case 3:
      density.density = candidate;
      break;
    case 4:
      population.qsh = candidate;
      break;
    case 5:
      population.qat = candidate;
      break;
    case 6:
      population.dipole = candidate;
      break;
    case 7:
      population.quadrupole = candidate;
      break;
    default:
      workspace.scratch = candidate;
      break;
  }
  const gpuxtb_status_t status = gpuxtb::detail::gfn2::evaluate_mulliken_population_cpu(
      fixture.plan, integrals, density, population, workspace, error);
  if (!all_equal_to(fixture.qsh, 71.0) || !all_equal_to(fixture.qat, 72.0) ||
      !all_equal_to(fixture.dipole, 73.0) || !all_equal_to(fixture.quadrupole, 74.0)) {
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  return status;
}

gpuxtb_status_t attempt_hamiltonian_control_alias(Fixture& fixture, double* candidate,
                                                  std::size_t buffer, std::string& error) {
  std::fill(fixture.hamiltonian.begin(), fixture.hamiltonian.end(), 81.0);
  MullikenIntegralView integrals = integral_view(fixture);
  MullikenPotentialView potential = potential_view(fixture);
  MullikenHamiltonianView hamiltonian = hamiltonian_view(fixture);
  MullikenWorkspace workspace = workspace_view(fixture);
  switch (buffer) {
    case 0:
      integrals.overlap = candidate;
      break;
    case 1:
      integrals.dipole = candidate;
      break;
    case 2:
      integrals.quadrupole = candidate;
      break;
    case 3:
      potential.vat = candidate;
      break;
    case 4:
      potential.vsh = candidate;
      break;
    case 5:
      potential.dipole = candidate;
      break;
    case 6:
      potential.quadrupole = candidate;
      break;
    case 7:
      hamiltonian.matrix = candidate;
      break;
    default:
      workspace.scratch = candidate;
      break;
  }
  const gpuxtb_status_t status = gpuxtb::detail::gfn2::add_mulliken_hamiltonian_cpu(
      fixture.plan, integrals, potential, hamiltonian, workspace, error);
  if (!all_equal_to(fixture.hamiltonian, 81.0)) {
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  return status;
}

gpuxtb_status_t attempt_population_descriptor_alias(Fixture& fixture, std::size_t control,
                                                    std::size_t byte_offset, std::size_t buffer,
                                                    std::string& error) {
  std::fill(fixture.qsh.begin(), fixture.qsh.end(), 91.0);
  std::fill(fixture.qat.begin(), fixture.qat.end(), 92.0);
  std::fill(fixture.dipole.begin(), fixture.dipole.end(), 93.0);
  std::fill(fixture.quadrupole.begin(), fixture.quadrupole.end(), 94.0);
  MullikenIntegralView integrals = integral_view(fixture);
  MullikenDensityView density = density_view(fixture);
  MullikenPopulationView population = population_view(fixture);
  MullikenWorkspace workspace = workspace_view(fixture);
  std::byte* control_address = nullptr;
  switch (control) {
    case 0:
      control_address = reinterpret_cast<std::byte*>(&integrals);
      break;
    case 1:
      control_address = reinterpret_cast<std::byte*>(&density);
      break;
    case 2:
      control_address = reinterpret_cast<std::byte*>(&population);
      break;
    default:
      control_address = reinterpret_cast<std::byte*>(&workspace);
      break;
  }
  auto* const candidate = reinterpret_cast<double*>(control_address + byte_offset);
  switch (buffer) {
    case 0:
      integrals.overlap = candidate;
      break;
    case 1:
      integrals.dipole = candidate;
      break;
    case 2:
      integrals.quadrupole = candidate;
      break;
    case 3:
      density.density = candidate;
      break;
    case 4:
      population.qsh = candidate;
      break;
    case 5:
      population.qat = candidate;
      break;
    case 6:
      population.dipole = candidate;
      break;
    case 7:
      population.quadrupole = candidate;
      break;
    default:
      workspace.scratch = candidate;
      break;
  }
  const auto integrals_before = object_bytes(integrals);
  const auto density_before = object_bytes(density);
  const auto population_before = object_bytes(population);
  const auto workspace_before = object_bytes(workspace);
  const gpuxtb_status_t status = gpuxtb::detail::gfn2::evaluate_mulliken_population_cpu(
      fixture.plan, integrals, density, population, workspace, error);
  if (!object_bytes_equal(integrals, integrals_before) ||
      !object_bytes_equal(density, density_before) ||
      !object_bytes_equal(population, population_before) ||
      !object_bytes_equal(workspace, workspace_before) || !all_equal_to(fixture.qsh, 91.0) ||
      !all_equal_to(fixture.qat, 92.0) || !all_equal_to(fixture.dipole, 93.0) ||
      !all_equal_to(fixture.quadrupole, 94.0)) {
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  return status;
}

gpuxtb_status_t attempt_hamiltonian_descriptor_alias(Fixture& fixture, std::size_t control,
                                                     std::size_t byte_offset, std::size_t buffer,
                                                     std::string& error) {
  std::fill(fixture.hamiltonian.begin(), fixture.hamiltonian.end(), 101.0);
  MullikenIntegralView integrals = integral_view(fixture);
  MullikenPotentialView potential = potential_view(fixture);
  MullikenHamiltonianView hamiltonian = hamiltonian_view(fixture);
  MullikenWorkspace workspace = workspace_view(fixture);
  std::byte* control_address = nullptr;
  switch (control) {
    case 0:
      control_address = reinterpret_cast<std::byte*>(&integrals);
      break;
    case 1:
      control_address = reinterpret_cast<std::byte*>(&potential);
      break;
    case 2:
      control_address = reinterpret_cast<std::byte*>(&hamiltonian);
      break;
    default:
      control_address = reinterpret_cast<std::byte*>(&workspace);
      break;
  }
  auto* const candidate = reinterpret_cast<double*>(control_address + byte_offset);
  switch (buffer) {
    case 0:
      integrals.overlap = candidate;
      break;
    case 1:
      integrals.dipole = candidate;
      break;
    case 2:
      integrals.quadrupole = candidate;
      break;
    case 3:
      potential.vat = candidate;
      break;
    case 4:
      potential.vsh = candidate;
      break;
    case 5:
      potential.dipole = candidate;
      break;
    case 6:
      potential.quadrupole = candidate;
      break;
    case 7:
      hamiltonian.matrix = candidate;
      break;
    default:
      workspace.scratch = candidate;
      break;
  }
  const auto integrals_before = object_bytes(integrals);
  const auto potential_before = object_bytes(potential);
  const auto hamiltonian_before = object_bytes(hamiltonian);
  const auto workspace_before = object_bytes(workspace);
  const gpuxtb_status_t status = gpuxtb::detail::gfn2::add_mulliken_hamiltonian_cpu(
      fixture.plan, integrals, potential, hamiltonian, workspace, error);
  if (!object_bytes_equal(integrals, integrals_before) ||
      !object_bytes_equal(potential, potential_before) ||
      !object_bytes_equal(hamiltonian, hamiltonian_before) ||
      !object_bytes_equal(workspace, workspace_before) ||
      !all_equal_to(fixture.hamiltonian, 101.0)) {
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  return status;
}

int test_two_ao_population_fixture() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 2}, {1, 1}, {0.0}, {0}, {1}, fixture, error));
  CHECK(fixture.plan.total_orbitals() == 2);
  CHECK(fixture.plan.total_shells() == 2);

  fixture.overlap = {1.0, 0.2, 0.3, 1.0};
  fixture.density = {0.8, 0.1, 0.4, 0.6};
  for (std::int64_t component = 0; component < 3; ++component) {
    const double scale = static_cast<double>(component + 1);
    const std::array<double, 4> block{scale, 2.0 * scale, 3.0 * scale, 4.0 * scale};
    std::copy(block.begin(), block.end(), fixture.dipole_integrals.begin() + component * 4);
  }
  for (std::int64_t component = 0; component < 6; ++component) {
    const double scale = static_cast<double>(component + 1);
    const std::array<double, 4> block{0.5 * scale, 1.5 * scale, 2.5 * scale, 3.5 * scale};
    std::copy(block.begin(), block.end(), fixture.quadrupole_integrals.begin() + component * 4);
  }

  CHECK(evaluate_population(fixture, error));
  CHECK(error.empty());
  CHECK(near(fixture.qsh[0], 0.08));
  CHECK(near(fixture.qsh[1], 0.38));
  CHECK(
      near(fixture.qsh[0] + fixture.qsh[1], 2.0 - (0.8 * 1.0 + 0.1 * 0.2 + 0.4 * 0.3 + 0.6 * 1.0)));
  CHECK(near(fixture.qat[0], fixture.qsh[0]));
  CHECK(near(fixture.qat[1], fixture.qsh[1]));
  for (std::int64_t component = 0; component < 3; ++component) {
    const double scale = static_cast<double>(component + 1);
    CHECK(near(fixture.dipole[static_cast<std::size_t>(component)], -2.0 * scale));
    CHECK(near(fixture.dipole[static_cast<std::size_t>(3 + component)], -2.6 * scale));
  }
  for (std::int64_t component = 0; component < 6; ++component) {
    const double scale = static_cast<double>(component + 1);
    CHECK(near(fixture.quadrupole[static_cast<std::size_t>(component)], -1.4 * scale));
    CHECK(near(fixture.quadrupole[static_cast<std::size_t>(6 + component)], -2.25 * scale));
  }
  return 0;
}

int test_multi_ao_shell_ownership() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 1}, {8}, {0.0}, {2}, {1}, fixture, error));
  CHECK(fixture.plan.total_shells() == 2);
  CHECK(fixture.plan.total_orbitals() == 4);
  std::fill(fixture.overlap.begin(), fixture.overlap.end(), 0.0);
  std::fill(fixture.density.begin(), fixture.density.end(), 0.0);
  std::fill(fixture.dipole_integrals.begin(), fixture.dipole_integrals.end(), 0.0);
  std::fill(fixture.quadrupole_integrals.begin(), fixture.quadrupole_integrals.end(), 0.0);
  for (std::size_t ket = 1; ket < 4u; ++ket) {
    fixture.density[ket] = 1.0;
    fixture.overlap[ket] = static_cast<double>(ket);
    fixture.dipole_integrals[ket] = static_cast<double>(ket + 3u);
  }
  CHECK(evaluate_population(fixture, error));
  CHECK(near(fixture.qsh[0], fixture.plan.reference_shell_occupations()[0]));
  CHECK(near(fixture.qsh[1], fixture.plan.reference_shell_occupations()[1] - 6.0));
  CHECK(near(fixture.qat[0], fixture.qsh[0] + fixture.qsh[1]));
  CHECK(near(fixture.dipole[0], -15.0));

  std::fill(fixture.overlap.begin(), fixture.overlap.end(), 0.0);
  for (std::size_t orbital = 1; orbital < 4u; ++orbital) {
    fixture.overlap[orbital * 4u + orbital] = 1.0;
  }
  std::fill(fixture.vat.begin(), fixture.vat.end(), 3.0);
  std::fill(fixture.vsh.begin(), fixture.vsh.end(), 0.0);
  fixture.vsh[1] = 2.0;
  std::fill(fixture.dipole_potential.begin(), fixture.dipole_potential.end(), 0.0);
  std::fill(fixture.quadrupole_potential.begin(), fixture.quadrupole_potential.end(), 0.0);
  std::fill(fixture.hamiltonian.begin(), fixture.hamiltonian.end(), 0.0);
  CHECK(assemble_hamiltonian(fixture, error));
  CHECK(fixture.hamiltonian[0] == 0.0);
  for (std::size_t orbital = 1; orbital < 4u; ++orbital) {
    CHECK(near(fixture.hamiltonian[orbital * 4u + orbital], -5.0));
  }
  return 0;
}

int test_hamiltonian_two_directions_and_packed_dual() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 2}, {1, 1}, {0.0}, {0}, {1}, fixture, error));
  std::fill(fixture.overlap.begin(), fixture.overlap.end(), 0.0);
  std::fill(fixture.dipole_integrals.begin(), fixture.dipole_integrals.end(), 0.0);
  std::fill(fixture.quadrupole_integrals.begin(), fixture.quadrupole_integrals.end(), 0.0);
  std::fill(fixture.hamiltonian.begin(), fixture.hamiltonian.end(), 0.0);
  fixture.overlap[1] = 0.4;
  fixture.overlap[2] = 0.4;
  fixture.vat = {1.0, 4.0};
  fixture.vsh = {2.0, 5.0};
  fixture.dipole_integrals[1] = 2.0;
  fixture.dipole_integrals[2] = 3.0;
  fixture.dipole_potential[0] = 10.0;
  fixture.dipole_potential[3] = 20.0;

  /* xy is packed component one and contracts as a dual without factor two. */
  fixture.quadrupole_integrals[4 + 1] = 7.0;
  fixture.quadrupole_integrals[4 + 2] = 11.0;
  fixture.quadrupole_potential[1] = 17.0;
  fixture.quadrupole_potential[6 + 1] = 13.0;
  CHECK(assemble_hamiltonian(fixture, error));

  const double scalar = -0.5 * 0.4 * (3.0 + 9.0);
  const double dipole = -0.5 * (2.0 * 20.0 + 3.0 * 10.0);
  const double quadrupole = -0.5 * (7.0 * 13.0 + 11.0 * 17.0);
  CHECK(near(fixture.hamiltonian[1], scalar + dipole + quadrupole));
  CHECK(near(fixture.hamiltonian[2], scalar + dipole + quadrupole));
  CHECK(fixture.hamiltonian[0] == 0.0);
  CHECK(fixture.hamiltonian[3] == 0.0);
  return 0;
}

int test_spin_swap() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 2}, {1, 1}, {0.0}, {0}, {2}, fixture, error));
  fixture.overlap = {1.0, 0.25, 0.25, 1.0};
  for (std::size_t index = 0; index < fixture.dipole_integrals.size(); ++index) {
    fixture.dipole_integrals[index] = 0.01 * static_cast<double>(index + 1u);
  }
  for (std::size_t index = 0; index < fixture.quadrupole_integrals.size(); ++index) {
    fixture.quadrupole_integrals[index] = -0.005 * static_cast<double>(index + 1u);
  }
  fixture.density = {0.7, 0.1, 0.1, 0.2, 0.2, -0.03, -0.03, 0.5};
  CHECK(evaluate_population(fixture, error));
  const std::vector<double> qsh = fixture.qsh;
  const std::vector<double> qat = fixture.qat;
  const std::vector<double> dipole = fixture.dipole;
  const std::vector<double> quadrupole = fixture.quadrupole;

  std::swap_ranges(fixture.density.begin(), fixture.density.begin() + 4,
                   fixture.density.begin() + 4);
  CHECK(evaluate_population(fixture, error));
  for (std::size_t shell = 0; shell < 2u; ++shell) {
    CHECK(near(fixture.qsh[shell], qsh[shell]));
    CHECK(near(fixture.qsh[2u + shell], -qsh[2u + shell]));
  }
  for (std::size_t atom = 0; atom < 2u; ++atom) {
    CHECK(near(fixture.qat[atom], qat[atom]));
    CHECK(near(fixture.qat[2u + atom], -qat[2u + atom]));
    for (std::size_t component = 0; component < 3u; ++component) {
      CHECK(near(fixture.dipole[atom * 3u + component], dipole[atom * 3u + component]));
      CHECK(near(fixture.dipole[(2u + atom) * 3u + component],
                 -dipole[(2u + atom) * 3u + component]));
    }
    for (std::size_t component = 0; component < 6u; ++component) {
      CHECK(near(fixture.quadrupole[atom * 6u + component], quadrupole[atom * 6u + component]));
      CHECK(near(fixture.quadrupole[(2u + atom) * 6u + component],
                 -quadrupole[(2u + atom) * 6u + component]));
    }
  }

  std::fill(fixture.overlap.begin(), fixture.overlap.end(), 0.0);
  std::fill(fixture.dipole_integrals.begin(), fixture.dipole_integrals.end(), 0.0);
  std::fill(fixture.quadrupole_integrals.begin(), fixture.quadrupole_integrals.end(), 0.0);
  std::fill(fixture.vat.begin(), fixture.vat.end(), 0.0);
  std::fill(fixture.vsh.begin(), fixture.vsh.end(), 0.0);
  std::fill(fixture.dipole_potential.begin(), fixture.dipole_potential.end(), 0.0);
  std::fill(fixture.quadrupole_potential.begin(), fixture.quadrupole_potential.end(), 0.0);
  fixture.overlap[0] = 1.0;
  fixture.vat[0] = 2.0;
  fixture.vat[2] = 0.6;
  std::fill(fixture.hamiltonian.begin(), fixture.hamiltonian.end(), 0.0);
  CHECK(assemble_hamiltonian(fixture, error));
  CHECK(near(fixture.hamiltonian[0], -1.3));
  CHECK(near(fixture.hamiltonian[4], -0.7));
  fixture.vat[2] = -0.6;
  std::fill(fixture.hamiltonian.begin(), fixture.hamiltonian.end(), 0.0);
  CHECK(assemble_hamiltonian(fixture, error));
  CHECK(near(fixture.hamiltonian[0], -0.7));
  CHECK(near(fixture.hamiltonian[4], -1.3));
  return 0;
}

int test_combined_adjoint_identity() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 2}, {1, 1}, {0.0}, {0}, {1}, fixture, error));
  fixture.overlap = {1.0, 0.3, 0.3, 1.0};
  fixture.density = {0.8, -0.2, -0.2, 0.5};
  for (std::size_t index = 0; index < fixture.dipole_integrals.size(); ++index) {
    fixture.dipole_integrals[index] = 0.07 * static_cast<double>(index + 1u) - 0.4;
  }
  for (std::size_t index = 0; index < fixture.quadrupole_integrals.size(); ++index) {
    fixture.quadrupole_integrals[index] = 0.03 * static_cast<double>(index + 2u) - 0.2;
  }
  fixture.vat = {0.4, -0.3};
  fixture.vsh = {-0.2, 0.7};
  fixture.dipole_potential = {0.1, -0.2, 0.3, -0.4, 0.5, -0.6};
  fixture.quadrupole_potential = {0.05,  -0.06, 0.07,  -0.08, 0.09,  -0.10,
                                  -0.11, 0.12,  -0.13, 0.14,  -0.15, 0.16};
  CHECK(evaluate_population(fixture, error));
  CHECK(assemble_hamiltonian(fixture, error));

  double lhs = 0.0;
  for (std::size_t index = 0; index < fixture.density.size(); ++index) {
    lhs += fixture.density[index] * fixture.hamiltonian[index];
  }
  double rhs = 0.0;
  for (std::size_t shell = 0; shell < fixture.qsh.size(); ++shell) {
    rhs += (fixture.qsh[shell] - fixture.plan.reference_shell_occupations()[shell]) *
           fixture.vsh[shell];
  }
  for (std::size_t atom = 0; atom < fixture.qat.size(); ++atom) {
    const double reference_atom = fixture.wavefunction.reference_atom_occupations[atom];
    rhs += (fixture.qat[atom] - reference_atom) * fixture.vat[atom];
  }
  for (std::size_t index = 0; index < fixture.dipole.size(); ++index) {
    rhs += fixture.dipole[index] * fixture.dipole_potential[index];
  }
  for (std::size_t index = 0; index < fixture.quadrupole.size(); ++index) {
    rhs += fixture.quadrupole[index] * fixture.quadrupole_potential[index];
  }
  CHECK(near(lhs, rhs, 5.0e-13));
  return 0;
}

int test_exact_plan_relationship_and_view_identity() {
  Fixture hydrogen;
  std::string error;
  CHECK(make_fixture({0, 2}, {1, 1}, {0.0}, {0}, {1}, hydrogen, error));
  const MullikenPlan original_plan = hydrogen.plan;
  const auto* const original_identity = hydrogen.plan.identity();

  WavefunctionLayout malformed_wavefunction = hydrogen.wavefunction;
  malformed_wavefunction.reference_shell_occupations[0] += 1.0;
  CHECK(gpuxtb::detail::gfn2::make_mulliken_plan(hydrogen.basis, hydrogen.integral_plan,
                                                 malformed_wavefunction, hydrogen.plan,
                                                 error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(hydrogen.plan.identity() == original_identity);

  malformed_wavefunction = hydrogen.wavefunction;
  malformed_wavefunction.reference_atom_occupations[1] += 0.25;
  CHECK(gpuxtb::detail::gfn2::make_mulliken_plan(hydrogen.basis, hydrogen.integral_plan,
                                                 malformed_wavefunction, hydrogen.plan,
                                                 error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(hydrogen.plan.identity() == original_identity);

  Fixture oxygen;
  CHECK(make_fixture({0, 1}, {8}, {0.0}, {2}, {2}, oxygen, error));
  malformed_wavefunction = oxygen.wavefunction;
  /* Carbon and oxygen have the same GFN2 shell/AO topology, so counts alone
   * cannot detect this stale-layout mismatch. */
  malformed_wavefunction.atomic_numbers[0] = 6;
  CHECK(gpuxtb::detail::gfn2::make_mulliken_plan(oxygen.basis, oxygen.integral_plan,
                                                 malformed_wavefunction, oxygen.plan,
                                                 error) == GPUXTB_STATUS_INVALID_ARGUMENT);

  Fixture ragged;
  CHECK(make_fixture({0, 2, 3}, {1, 1, 8}, {0.0, 0.0}, {0, 2}, {1, 2}, ragged, error));
  const auto* const ragged_identity = ragged.plan.identity();
  BasisPlan malformed_basis = ragged.basis;
  malformed_basis.batch_shell_offsets[1] += 1;
  CHECK(gpuxtb::detail::gfn2::make_mulliken_plan(malformed_basis, ragged.integral_plan,
                                                 ragged.wavefunction, ragged.plan,
                                                 error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(ragged.plan.identity() == ragged_identity);

  IntegralPlan malformed_integrals = ragged.integral_plan;
  malformed_integrals.matrix_offsets[1] += 1;
  CHECK(gpuxtb::detail::gfn2::make_mulliken_plan(ragged.basis, malformed_integrals,
                                                 ragged.wavefunction, ragged.plan,
                                                 error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(ragged.plan.identity() == ragged_identity);

  malformed_wavefunction = ragged.wavefunction;
  malformed_wavefunction.density.system_offsets[1] += 1;
  CHECK(gpuxtb::detail::gfn2::make_mulliken_plan(ragged.basis, ragged.integral_plan,
                                                 malformed_wavefunction, ragged.plan,
                                                 error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(ragged.plan.identity() == ragged_identity);

  Fixture reverse_ragged;
  CHECK(make_fixture({0, 1, 3}, {8, 1, 1}, {0.0, 0.0}, {2, 0}, {2, 1}, reverse_ragged, error));
  CHECK(reverse_ragged.plan.matrix_elements() == ragged.plan.matrix_elements());
  CHECK(reverse_ragged.plan.density_elements() == ragged.plan.density_elements());
  CHECK(reverse_ragged.plan.matrix_offsets() != ragged.plan.matrix_offsets());

  MullikenIntegralView wrong_integrals = integral_view(ragged);
  wrong_integrals.plan_identity = reverse_ragged.plan.identity();
  CHECK(gpuxtb::detail::gfn2::evaluate_mulliken_population_cpu(
            ragged.plan, wrong_integrals, density_view(ragged), population_view(ragged),
            workspace_view(ragged), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  MullikenDensityView wrong_density = density_view(ragged);
  wrong_density.plan_identity = reverse_ragged.plan.identity();
  CHECK(gpuxtb::detail::gfn2::evaluate_mulliken_population_cpu(
            ragged.plan, integral_view(ragged), wrong_density, population_view(ragged),
            workspace_view(ragged), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  MullikenPopulationView wrong_population = population_view(ragged);
  wrong_population.plan_identity = reverse_ragged.plan.identity();
  CHECK(gpuxtb::detail::gfn2::evaluate_mulliken_population_cpu(
            ragged.plan, integral_view(ragged), density_view(ragged), wrong_population,
            workspace_view(ragged), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  MullikenPotentialView wrong_potential = potential_view(ragged);
  wrong_potential.plan_identity = reverse_ragged.plan.identity();
  CHECK(gpuxtb::detail::gfn2::add_mulliken_hamiltonian_cpu(
            ragged.plan, integral_view(ragged), wrong_potential, hamiltonian_view(ragged),
            workspace_view(ragged), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  MullikenHamiltonianView wrong_hamiltonian = hamiltonian_view(ragged);
  wrong_hamiltonian.plan_identity = reverse_ragged.plan.identity();
  CHECK(gpuxtb::detail::gfn2::add_mulliken_hamiltonian_cpu(
            ragged.plan, integral_view(ragged), potential_view(ragged), wrong_hamiltonian,
            workspace_view(ragged), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  hydrogen.plan = original_plan;
  return 0;
}

int test_pinned_tblite_open_shell_literal_oracle() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 2}, {1, 1}, {0.0}, {0}, {2}, fixture, error));

  /* Literal oracle independently evaluated from tblite revision
   * e9abc395b122018ed688aecb1c3a65cecaf97beb, specifically
   * src/tblite/wavefunction/mulliken.f90, src/tblite/scf/potential.f90, and
   * src/tblite/wavefunction/spin.f90. The deliberately nonsymmetric random
   * density catches row/column ownership errors; expected values are not
   * generated by the implementation under test. */
  fixture.overlap = {1.0, 0.2, 0.2, 1.0};
  fixture.density = {0.7, 0.11, -0.04, 0.2, 0.25, -0.03, 0.08, 0.55};
  fixture.dipole_integrals = {-0.145, -0.12, -0.095, -0.07, -0.045, -0.02,
                              0.005,  0.03,  0.055,  0.08,  0.105,  0.13};
  fixture.quadrupole_integrals = {0.079,  0.068,  0.057,  0.046,  0.035,  0.024,  0.013,  0.002,
                                  -0.009, -0.02,  -0.031, -0.042, -0.053, -0.064, -0.075, -0.086,
                                  -0.097, -0.108, -0.119, -0.13,  -0.141, -0.152, -0.163, -0.174};
  CHECK(evaluate_population(fixture, error));
  CHECK(all_near(fixture.qsh, {0.042, 0.234, -0.426, 0.322}));
  CHECK(all_near(fixture.qat, fixture.qsh));
  CHECK(all_near(fixture.dipole, {0.14155, 0.04255, -0.05645, 0.0621, -0.0209, -0.1039, 0.05385,
                                  0.02085, -0.01215, -0.0077, 0.0133, 0.0343}));
  CHECK(all_near(fixture.quadrupole,
                 {-0.07733, -0.03377, 0.00979, 0.05335,  0.09691,  0.14047,  -0.03994, -0.00342,
                  0.0331,   0.06962,  0.10614, 0.14266,  -0.02871, -0.01419, 0.00033,  0.01485,
                  0.02937,  0.04389,  0.00658, -0.00266, -0.0119,  -0.02114, -0.03038, -0.03962}));

  fixture.vat = {0.4, -0.2, 0.1, 0.3};
  fixture.vsh = {-0.15, 0.25, -0.05, 0.12};
  fixture.dipole_potential = {-0.063, -0.046, -0.029, -0.012, 0.005, 0.022,
                              0.039,  0.056,  0.073,  0.09,   0.107, 0.124};
  fixture.quadrupole_potential = {0.101,  0.092,  0.083,  0.074, 0.065,  0.056,  0.047,  0.038,
                                  0.029,  0.02,   0.011,  0.002, -0.007, -0.016, -0.025, -0.034,
                                  -0.043, -0.052, -0.061, -0.07, -0.079, -0.088, -0.097, -0.106};
  fixture.hamiltonian = {0.02, -0.01, 0.03, 0.04, -0.02, 0.05, 0.01, -0.03};
  CHECK(assemble_hamiltonian(fixture, error));
  CHECK(all_near(fixture.hamiltonian, {-0.135098, -0.057009, -0.017009, -0.221698, -0.116841,
                                       0.0727425, 0.0327425, 0.180326}));
  return 0;
}

int test_open_shell_adjoint_identity() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 2}, {1, 1}, {0.0}, {0}, {2}, fixture, error));
  fixture.overlap = {1.0, 0.31, 0.31, 1.0};
  /* The Hamiltonian is symmetrized, so the adjoint identity requires a
   * physical symmetric density in each spin channel. Nonsymmetric ownership
   * is covered separately by the pinned tblite literal oracle above. */
  fixture.density = {0.61, -0.13, -0.13, 0.22, 0.17, 0.04, 0.04, 0.48};
  for (std::size_t index = 0; index < fixture.dipole_integrals.size(); ++index) {
    fixture.dipole_integrals[index] = 0.023 * static_cast<double>(index + 1u) - 0.19;
  }
  for (std::size_t index = 0; index < fixture.quadrupole_integrals.size(); ++index) {
    fixture.quadrupole_integrals[index] = -0.013 * static_cast<double>(index + 2u) + 0.16;
  }
  for (std::size_t index = 0; index < fixture.vat.size(); ++index) {
    fixture.vat[index] = 0.08 * static_cast<double>(index + 1u) - 0.21;
  }
  for (std::size_t index = 0; index < fixture.vsh.size(); ++index) {
    fixture.vsh[index] = -0.06 * static_cast<double>(index + 1u) + 0.17;
  }
  for (std::size_t index = 0; index < fixture.dipole_potential.size(); ++index) {
    fixture.dipole_potential[index] = 0.015 * static_cast<double>(index + 1u) - 0.09;
  }
  for (std::size_t index = 0; index < fixture.quadrupole_potential.size(); ++index) {
    fixture.quadrupole_potential[index] = -0.007 * static_cast<double>(index + 1u) + 0.12;
  }
  std::fill(fixture.hamiltonian.begin(), fixture.hamiltonian.end(), 0.0);
  CHECK(evaluate_population(fixture, error));
  CHECK(assemble_hamiltonian(fixture, error));

  double lhs = 0.0;
  for (std::size_t index = 0; index < fixture.density.size(); ++index) {
    lhs += fixture.density[index] * fixture.hamiltonian[index];
  }
  double rhs = 0.0;
  for (std::size_t shell = 0; shell < 2u; ++shell) {
    rhs += (fixture.qsh[shell] - fixture.plan.reference_shell_occupations()[shell]) *
           fixture.vsh[shell];
    rhs += fixture.qsh[2u + shell] * fixture.vsh[2u + shell];
  }
  for (std::size_t atom = 0; atom < 2u; ++atom) {
    rhs += (fixture.qat[atom] - fixture.wavefunction.reference_atom_occupations[atom]) *
           fixture.vat[atom];
    rhs += fixture.qat[2u + atom] * fixture.vat[2u + atom];
  }
  for (std::size_t index = 0; index < fixture.dipole.size(); ++index) {
    rhs += fixture.dipole[index] * fixture.dipole_potential[index];
  }
  for (std::size_t index = 0; index < fixture.quadrupole.size(); ++index) {
    rhs += fixture.quadrupole[index] * fixture.quadrupole_potential[index];
  }
  CHECK(near(lhs, 0.5 * rhs, 8.0e-13));
  return 0;
}

int test_batch_matches_sequential() {
  Fixture batch;
  std::string error;
  CHECK(make_fixture({0, 2, 4}, {1, 1, 1, 1}, {0.0, 0.0}, {0, 0}, {1, 1}, batch, error));
  CHECK(batch.plan.matrix_elements() == 8);
  for (std::size_t index = 0; index < batch.overlap.size(); ++index) {
    batch.overlap[index] = 0.1 + 0.02 * static_cast<double>(index);
  }
  for (std::size_t index = 0; index < batch.dipole_integrals.size(); ++index) {
    batch.dipole_integrals[index] = -0.3 + 0.01 * static_cast<double>(index);
  }
  for (std::size_t index = 0; index < batch.quadrupole_integrals.size(); ++index) {
    batch.quadrupole_integrals[index] = 0.2 - 0.004 * static_cast<double>(index);
  }
  for (std::size_t index = 0; index < batch.density.size(); ++index) {
    batch.density[index] = 0.05 * static_cast<double>(index + 1u);
  }
  CHECK(evaluate_population(batch, error));
  for (std::size_t index = 0; index < batch.vat.size(); ++index) {
    batch.vat[index] = -0.2 + 0.1 * static_cast<double>(index);
    batch.vsh[index] = 0.3 - 0.04 * static_cast<double>(index);
  }
  for (std::size_t index = 0; index < batch.dipole_potential.size(); ++index) {
    batch.dipole_potential[index] = -0.1 + 0.02 * static_cast<double>(index);
  }
  for (std::size_t index = 0; index < batch.quadrupole_potential.size(); ++index) {
    batch.quadrupole_potential[index] = 0.08 - 0.006 * static_cast<double>(index);
  }
  for (std::size_t index = 0; index < batch.hamiltonian.size(); ++index) {
    batch.hamiltonian[index] = 0.01 * static_cast<double>(index + 1u);
  }
  CHECK(assemble_hamiltonian(batch, error));

  for (std::size_t system = 0; system < 2u; ++system) {
    Fixture sequential;
    CHECK(make_fixture({0, 2}, {1, 1}, {0.0}, {0}, {1}, sequential, error));
    std::copy_n(batch.overlap.begin() + static_cast<std::ptrdiff_t>(system * 4u), 4u,
                sequential.overlap.begin());
    for (std::size_t component = 0; component < 3u; ++component) {
      std::copy_n(
          batch.dipole_integrals.begin() +
              static_cast<std::ptrdiff_t>(component * 8u + system * 4u),
          4u, sequential.dipole_integrals.begin() + static_cast<std::ptrdiff_t>(component * 4u));
    }
    for (std::size_t component = 0; component < 6u; ++component) {
      std::copy_n(
          batch.quadrupole_integrals.begin() +
              static_cast<std::ptrdiff_t>(component * 8u + system * 4u),
          4u,
          sequential.quadrupole_integrals.begin() + static_cast<std::ptrdiff_t>(component * 4u));
    }
    std::copy_n(batch.density.begin() + static_cast<std::ptrdiff_t>(system * 4u), 4u,
                sequential.density.begin());
    std::copy_n(batch.vat.begin() + static_cast<std::ptrdiff_t>(system * 2u), 2u,
                sequential.vat.begin());
    std::copy_n(batch.vsh.begin() + static_cast<std::ptrdiff_t>(system * 2u), 2u,
                sequential.vsh.begin());
    std::copy_n(batch.dipole_potential.begin() + static_cast<std::ptrdiff_t>(system * 6u), 6u,
                sequential.dipole_potential.begin());
    std::copy_n(batch.quadrupole_potential.begin() + static_cast<std::ptrdiff_t>(system * 12u), 12u,
                sequential.quadrupole_potential.begin());
    /* Reconstruct the pre-assembly H used by this system. */
    for (std::size_t index = 0; index < 4u; ++index) {
      sequential.hamiltonian[index] = 0.01 * static_cast<double>(system * 4u + index + 1u);
    }
    CHECK(evaluate_population(sequential, error));
    CHECK(assemble_hamiltonian(sequential, error));
    CHECK(std::equal(sequential.qsh.begin(), sequential.qsh.end(),
                     batch.qsh.begin() + static_cast<std::ptrdiff_t>(system * 2u)));
    CHECK(std::equal(sequential.qat.begin(), sequential.qat.end(),
                     batch.qat.begin() + static_cast<std::ptrdiff_t>(system * 2u)));
    CHECK(std::equal(sequential.dipole.begin(), sequential.dipole.end(),
                     batch.dipole.begin() + static_cast<std::ptrdiff_t>(system * 6u)));
    CHECK(std::equal(sequential.quadrupole.begin(), sequential.quadrupole.end(),
                     batch.quadrupole.begin() + static_cast<std::ptrdiff_t>(system * 12u)));
    CHECK(std::equal(sequential.hamiltonian.begin(), sequential.hamiltonian.end(),
                     batch.hamiltonian.begin() + static_cast<std::ptrdiff_t>(system * 4u)));
  }
  return 0;
}

int test_mixed_spin_heterogeneous_ragged_batch() {
  Fixture batch;
  std::string error;
  CHECK(make_fixture({0, 2, 3}, {1, 1, 8}, {0.0, 0.0}, {0, 2}, {1, 2}, batch, error));
  CHECK(batch.plan.matrix_offsets()[1] == 4);
  CHECK(batch.wavefunction.density.system_offsets[1] == 4);
  for (std::size_t index = 0; index < batch.overlap.size(); ++index) {
    batch.overlap[index] = 0.015 * static_cast<double>(index + 1u) - 0.12;
  }
  for (std::size_t index = 0; index < batch.dipole_integrals.size(); ++index) {
    batch.dipole_integrals[index] = -0.006 * static_cast<double>(index + 2u) + 0.21;
  }
  for (std::size_t index = 0; index < batch.quadrupole_integrals.size(); ++index) {
    batch.quadrupole_integrals[index] = 0.0025 * static_cast<double>(index + 3u) - 0.14;
  }
  for (std::size_t index = 0; index < batch.density.size(); ++index) {
    batch.density[index] = -0.009 * static_cast<double>(index + 1u) + 0.37;
  }
  for (std::size_t index = 0; index < batch.vat.size(); ++index) {
    batch.vat[index] = 0.031 * static_cast<double>(index + 1u) - 0.18;
  }
  for (std::size_t index = 0; index < batch.vsh.size(); ++index) {
    batch.vsh[index] = -0.027 * static_cast<double>(index + 2u) + 0.16;
  }
  for (std::size_t index = 0; index < batch.dipole_potential.size(); ++index) {
    batch.dipole_potential[index] = 0.004 * static_cast<double>(index + 1u) - 0.07;
  }
  for (std::size_t index = 0; index < batch.quadrupole_potential.size(); ++index) {
    batch.quadrupole_potential[index] = -0.003 * static_cast<double>(index + 1u) + 0.11;
  }
  for (std::size_t index = 0; index < batch.hamiltonian.size(); ++index) {
    batch.hamiltonian[index] = 0.001 * static_cast<double>(index + 1u);
  }
  const std::vector<double> initial_hamiltonian = batch.hamiltonian;
  CHECK(evaluate_population(batch, error));
  CHECK(assemble_hamiltonian(batch, error));

  Fixture hydrogen;
  Fixture oxygen;
  CHECK(make_fixture({0, 2}, {1, 1}, {0.0}, {0}, {1}, hydrogen, error));
  CHECK(make_fixture({0, 1}, {8}, {0.0}, {2}, {2}, oxygen, error));
  const std::array<Fixture*, 2> sequential_fixtures{&hydrogen, &oxygen};
  for (std::size_t system = 0; system < sequential_fixtures.size(); ++system) {
    Fixture& sequential = *sequential_fixtures[system];
    const std::int64_t matrix_begin = batch.integral_plan.matrix_offsets[system];
    const std::int64_t matrix_end = batch.integral_plan.matrix_offsets[system + 1u];
    const std::int64_t matrix_elements = matrix_end - matrix_begin;
    std::copy_n(batch.overlap.begin() + matrix_begin, matrix_elements, sequential.overlap.begin());
    for (std::int64_t component = 0; component < 3; ++component) {
      std::copy_n(
          batch.dipole_integrals.begin() + component * batch.plan.matrix_elements() + matrix_begin,
          matrix_elements,
          sequential.dipole_integrals.begin() + component * sequential.plan.matrix_elements());
    }
    for (std::int64_t component = 0; component < 6; ++component) {
      std::copy_n(
          batch.quadrupole_integrals.begin() + component * batch.plan.matrix_elements() +
              matrix_begin,
          matrix_elements,
          sequential.quadrupole_integrals.begin() + component * sequential.plan.matrix_elements());
    }

    const auto copy_system_field =
        [system](const std::vector<double>& source,
                 const gpuxtb::detail::gfn2::WavefunctionFieldLayout& source_layout,
                 std::vector<double>& destination) {
          const std::int64_t begin = source_layout.system_offsets[system];
          std::copy_n(source.begin() + begin, static_cast<std::int64_t>(destination.size()),
                      destination.begin());
        };
    copy_system_field(batch.density, batch.wavefunction.density, sequential.density);
    copy_system_field(batch.vat, batch.wavefunction.qat, sequential.vat);
    copy_system_field(batch.vsh, batch.wavefunction.qsh, sequential.vsh);
    copy_system_field(batch.dipole_potential, batch.wavefunction.dipole,
                      sequential.dipole_potential);
    copy_system_field(batch.quadrupole_potential, batch.wavefunction.quadrupole,
                      sequential.quadrupole_potential);
    copy_system_field(initial_hamiltonian, batch.wavefunction.density, sequential.hamiltonian);
    CHECK(evaluate_population(sequential, error));
    CHECK(assemble_hamiltonian(sequential, error));

    const auto matches_batch_field =
        [system](const std::vector<double>& expected, const std::vector<double>& actual,
                 const gpuxtb::detail::gfn2::WavefunctionFieldLayout& batch_layout) {
          const std::int64_t begin = batch_layout.system_offsets[system];
          return std::equal(expected.begin(), expected.end(), actual.begin() + begin);
        };
    CHECK(matches_batch_field(sequential.qsh, batch.qsh, batch.wavefunction.qsh));
    CHECK(matches_batch_field(sequential.qat, batch.qat, batch.wavefunction.qat));
    CHECK(matches_batch_field(sequential.dipole, batch.dipole, batch.wavefunction.dipole));
    CHECK(matches_batch_field(sequential.quadrupole, batch.quadrupole,
                              batch.wavefunction.quadrupole));
    CHECK(
        matches_batch_field(sequential.hamiltonian, batch.hamiltonian, batch.wavefunction.density));
  }
  return 0;
}

int test_translation_invariance_with_integral_evaluator() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 2}, {1, 1}, {0.0}, {0}, {1}, fixture, error));
  const std::array<double, 6> positions{0.1, -0.2, 0.3, 1.4, 0.5, -0.7};
  std::array<double, 6> translated = positions;
  for (std::size_t atom = 0; atom < 2u; ++atom) {
    translated[atom * 3u] += 4.2;
    translated[atom * 3u + 1u] -= 3.1;
    translated[atom * 3u + 2u] += 2.7;
  }
  std::vector<double> integral_workspace(
      (fixture.integral_plan.workspace_size_bytes + sizeof(double) - 1u) / sizeof(double));
  CHECK(gpuxtb::detail::gfn2::evaluate_overlap_cpu(
            fixture.basis, fixture.integral_plan, positions.data(), fixture.overlap.data(),
            integral_workspace.data(), integral_workspace.size() * sizeof(double),
            error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::evaluate_multipole_cpu(
            fixture.basis, fixture.integral_plan, positions.data(), fixture.dipole_integrals.data(),
            fixture.quadrupole_integrals.data(), integral_workspace.data(),
            integral_workspace.size() * sizeof(double), error) == GPUXTB_STATUS_SUCCESS);
  fixture.density = {0.7, 0.2, 0.2, 0.6};
  CHECK(evaluate_population(fixture, error));
  const std::vector<double> qsh = fixture.qsh;
  const std::vector<double> dipole = fixture.dipole;
  const std::vector<double> quadrupole = fixture.quadrupole;

  CHECK(gpuxtb::detail::gfn2::evaluate_overlap_cpu(
            fixture.basis, fixture.integral_plan, translated.data(), fixture.overlap.data(),
            integral_workspace.data(), integral_workspace.size() * sizeof(double),
            error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::evaluate_multipole_cpu(
            fixture.basis, fixture.integral_plan, translated.data(),
            fixture.dipole_integrals.data(), fixture.quadrupole_integrals.data(),
            integral_workspace.data(), integral_workspace.size() * sizeof(double),
            error) == GPUXTB_STATUS_SUCCESS);
  CHECK(evaluate_population(fixture, error));
  for (std::size_t index = 0; index < qsh.size(); ++index) {
    CHECK(near(fixture.qsh[index], qsh[index], 5.0e-13));
  }
  for (std::size_t index = 0; index < dipole.size(); ++index) {
    CHECK(near(fixture.dipole[index], dipole[index], 5.0e-13));
  }
  for (std::size_t index = 0; index < quadrupole.size(); ++index) {
    CHECK(near(fixture.quadrupole[index], quadrupole[index], 1.0e-12));
  }

  std::array<double, 6> rotated{};
  for (std::size_t atom = 0; atom < 2u; ++atom) {
    /* Ninety-degree active rotation about z: (x,y,z) -> (-y,x,z). */
    rotated[atom * 3u] = -positions[atom * 3u + 1u];
    rotated[atom * 3u + 1u] = positions[atom * 3u];
    rotated[atom * 3u + 2u] = positions[atom * 3u + 2u];
  }
  CHECK(gpuxtb::detail::gfn2::evaluate_overlap_cpu(
            fixture.basis, fixture.integral_plan, rotated.data(), fixture.overlap.data(),
            integral_workspace.data(), integral_workspace.size() * sizeof(double),
            error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::evaluate_multipole_cpu(
            fixture.basis, fixture.integral_plan, rotated.data(), fixture.dipole_integrals.data(),
            fixture.quadrupole_integrals.data(), integral_workspace.data(),
            integral_workspace.size() * sizeof(double), error) == GPUXTB_STATUS_SUCCESS);
  CHECK(evaluate_population(fixture, error));
  for (std::size_t atom = 0; atom < 2u; ++atom) {
    CHECK(near(fixture.dipole[atom * 3u], -dipole[atom * 3u + 1u], 1.0e-12));
    CHECK(near(fixture.dipole[atom * 3u + 1u], dipole[atom * 3u], 1.0e-12));
    CHECK(near(fixture.dipole[atom * 3u + 2u], dipole[atom * 3u + 2u], 1.0e-12));
    const std::size_t offset = atom * 6u;
    CHECK(near(fixture.quadrupole[offset], quadrupole[offset + 2u], 2.0e-12));
    CHECK(near(fixture.quadrupole[offset + 1u], -quadrupole[offset + 1u], 2.0e-12));
    CHECK(near(fixture.quadrupole[offset + 2u], quadrupole[offset], 2.0e-12));
    CHECK(near(fixture.quadrupole[offset + 3u], -quadrupole[offset + 4u], 2.0e-12));
    CHECK(near(fixture.quadrupole[offset + 4u], quadrupole[offset + 3u], 2.0e-12));
    CHECK(near(fixture.quadrupole[offset + 5u], quadrupole[offset + 5u], 2.0e-12));
  }
  return 0;
}

int test_alias_atomicity_and_zero_allocation() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 2, 4}, {1, 1, 1, 1}, {0.0, 0.0}, {0, 0}, {1, 1}, fixture, error));
  std::fill(fixture.overlap.begin(), fixture.overlap.end(), 0.2);
  std::fill(fixture.dipole_integrals.begin(), fixture.dipole_integrals.end(), 0.1);
  std::fill(fixture.quadrupole_integrals.begin(), fixture.quadrupole_integrals.end(), -0.05);
  std::fill(fixture.density.begin(), fixture.density.end(), 0.3);
  std::fill(fixture.qsh.begin(), fixture.qsh.end(), 41.0);
  std::fill(fixture.qat.begin(), fixture.qat.end(), 42.0);
  std::fill(fixture.dipole.begin(), fixture.dipole.end(), 43.0);
  std::fill(fixture.quadrupole.begin(), fixture.quadrupole.end(), 44.0);

  MullikenPopulationView aliased = population_view(fixture);
  aliased.qat = fixture.qsh.data();
  CHECK(gpuxtb::detail::gfn2::evaluate_mulliken_population_cpu(
            fixture.plan, integral_view(fixture), density_view(fixture), aliased,
            workspace_view(fixture), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(all_equal_to(fixture.qsh, 41.0));
  CHECK(all_equal_to(fixture.qat, 42.0));

  fixture.density[6] = std::numeric_limits<double>::infinity();
  CHECK(gpuxtb::detail::gfn2::evaluate_mulliken_population_cpu(
            fixture.plan, integral_view(fixture), density_view(fixture), population_view(fixture),
            workspace_view(fixture), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(fixture.dipole.begin(), fixture.dipole.end(),
                    [](double value) { return value == 43.0; }));
  fixture.density[6] = 0.3;
  fixture.density[6] = std::numeric_limits<double>::max();
  fixture.overlap[6] = 2.0;
  CHECK(gpuxtb::detail::gfn2::evaluate_mulliken_population_cpu(
            fixture.plan, integral_view(fixture), density_view(fixture), population_view(fixture),
            workspace_view(fixture), error) == GPUXTB_STATUS_INTERNAL_ERROR);
  CHECK(std::all_of(fixture.quadrupole.begin(), fixture.quadrupole.end(),
                    [](double value) { return value == 44.0; }));

  fixture.density[6] = 0.3;
  fixture.overlap[6] = 0.2;
  std::fill(fixture.hamiltonian.begin(), fixture.hamiltonian.end(), 17.0);
  MullikenIntegralView h_alias = integral_view(fixture);
  h_alias.overlap = fixture.hamiltonian.data();
  CHECK(gpuxtb::detail::gfn2::add_mulliken_hamiltonian_cpu(
            fixture.plan, h_alias, potential_view(fixture), hamiltonian_view(fixture),
            workspace_view(fixture), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(fixture.hamiltonian.begin(), fixture.hamiltonian.end(),
                    [](double value) { return value == 17.0; }));

  fixture.overlap[4] = 2.0;
  fixture.vat[2] = std::numeric_limits<double>::max();
  CHECK(gpuxtb::detail::gfn2::add_mulliken_hamiltonian_cpu(
            fixture.plan, integral_view(fixture), potential_view(fixture),
            hamiltonian_view(fixture), workspace_view(fixture),
            error) == GPUXTB_STATUS_INTERNAL_ERROR);
  CHECK(std::all_of(fixture.hamiltonian.begin(), fixture.hamiltonian.end(),
                    [](double value) { return value == 17.0; }));
  fixture.overlap[4] = 0.2;
  fixture.vat[2] = 0.0;

  std::fill(fixture.vat.begin(), fixture.vat.end(), 0.1);
  std::fill(fixture.vsh.begin(), fixture.vsh.end(), -0.2);
  std::fill(fixture.dipole_potential.begin(), fixture.dipole_potential.end(), 0.05);
  std::fill(fixture.quadrupole_potential.begin(), fixture.quadrupole_potential.end(), -0.03);
  allocation_test::count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const gpuxtb_status_t population_status = gpuxtb::detail::gfn2::evaluate_mulliken_population_cpu(
      fixture.plan, integral_view(fixture), density_view(fixture), population_view(fixture),
      workspace_view(fixture), error);
  const gpuxtb_status_t hamiltonian_status = gpuxtb::detail::gfn2::add_mulliken_hamiltonian_cpu(
      fixture.plan, integral_view(fixture), potential_view(fixture), hamiltonian_view(fixture),
      workspace_view(fixture), error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  CHECK(population_status == GPUXTB_STATUS_SUCCESS);
  CHECK(hamiltonian_status == GPUXTB_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == 0u);
  return 0;
}

int test_control_storage_alias_matrix() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 2}, {1, 1}, {0.0}, {0}, {1}, fixture, error));
  std::fill(fixture.overlap.begin(), fixture.overlap.end(), 0.2);
  std::fill(fixture.dipole_integrals.begin(), fixture.dipole_integrals.end(), 0.1);
  std::fill(fixture.quadrupole_integrals.begin(), fixture.quadrupole_integrals.end(), -0.05);
  std::fill(fixture.density.begin(), fixture.density.end(), 0.3);
  std::fill(fixture.vat.begin(), fixture.vat.end(), 0.11);
  std::fill(fixture.vsh.begin(), fixture.vsh.end(), -0.12);
  std::fill(fixture.dipole_potential.begin(), fixture.dipole_potential.end(), 0.07);
  std::fill(fixture.quadrupole_potential.begin(), fixture.quadrupole_potential.end(), -0.03);

  const auto* const identity = fixture.plan.identity();
  const auto plan_bytes = object_bytes(fixture.plan);
  const std::vector<std::int64_t> atom_offsets = fixture.plan.atom_offsets();
  const std::vector<std::int64_t> shell_offsets = fixture.plan.batch_shell_offsets();
  const std::vector<std::int64_t> orbital_offsets = fixture.plan.batch_orbital_offsets();
  const std::vector<std::int64_t> matrix_offsets = fixture.plan.matrix_offsets();
  const std::vector<std::int64_t> shell_orbital_offsets = fixture.plan.shell_orbital_offsets();
  const std::vector<std::int64_t> shell_to_atom = fixture.plan.shell_to_atom();
  const std::vector<std::int64_t> orbital_to_shell = fixture.plan.orbital_to_shell();
  const std::vector<std::int64_t> orbital_to_atom = fixture.plan.orbital_to_atom();
  const std::vector<std::int32_t> spin_channels = fixture.plan.spin_channels();
  const std::vector<double> references = fixture.plan.reference_shell_occupations();
  const auto plan_unchanged = [&]() {
    return fixture.plan.identity() == identity && object_bytes_equal(fixture.plan, plan_bytes) &&
           fixture.plan.atom_offsets() == atom_offsets &&
           fixture.plan.batch_shell_offsets() == shell_offsets &&
           fixture.plan.batch_orbital_offsets() == orbital_offsets &&
           fixture.plan.matrix_offsets() == matrix_offsets &&
           fixture.plan.shell_orbital_offsets() == shell_orbital_offsets &&
           fixture.plan.shell_to_atom() == shell_to_atom &&
           fixture.plan.orbital_to_shell() == orbital_to_shell &&
           fixture.plan.orbital_to_atom() == orbital_to_atom &&
           fixture.plan.spin_channels() == spin_channels &&
           fixture.plan.reference_shell_occupations() == references;
  };

  const auto* plan_address = reinterpret_cast<const std::byte*>(&fixture.plan);
  const auto* data_address = reinterpret_cast<const std::byte*>(fixture.plan.identity());
  const auto* reference_address = reinterpret_cast<const std::byte*>(references.data());
  const std::array<const std::byte*, 6> exact_and_partial_controls{
      plan_address,
      plan_address + sizeof(double),
      data_address,
      data_address + sizeof(double),
      reinterpret_cast<const std::byte*>(fixture.plan.reference_shell_occupations().data()),
      reinterpret_cast<const std::byte*>(fixture.plan.reference_shell_occupations().data()) +
          sizeof(double)};
  CHECK(reference_address != nullptr);
  for (const std::byte* control : exact_and_partial_controls) {
    auto* const candidate = reinterpret_cast<double*>(const_cast<std::byte*>(control));
    for (std::size_t buffer = 0; buffer < 9u; ++buffer) {
      CHECK(attempt_population_control_alias(fixture, candidate, buffer, error) ==
            GPUXTB_STATUS_INVALID_ARGUMENT);
      CHECK(plan_unchanged());
      CHECK(attempt_hamiltonian_control_alias(fixture, candidate, buffer, error) ==
            GPUXTB_STATUS_INVALID_ARGUMENT);
      CHECK(plan_unchanged());
    }
  }

  const std::array<const void*, 10> vector_backings{
      fixture.plan.atom_offsets().data(),
      fixture.plan.batch_shell_offsets().data(),
      fixture.plan.batch_orbital_offsets().data(),
      fixture.plan.matrix_offsets().data(),
      fixture.plan.shell_orbital_offsets().data(),
      fixture.plan.shell_to_atom().data(),
      fixture.plan.orbital_to_shell().data(),
      fixture.plan.orbital_to_atom().data(),
      fixture.plan.spin_channels().data(),
      fixture.plan.reference_shell_occupations().data()};
  for (const void* control : vector_backings) {
    auto* const candidate = reinterpret_cast<double*>(const_cast<void*>(control));
    for (std::size_t buffer = 0; buffer < 9u; ++buffer) {
      CHECK(attempt_population_control_alias(fixture, candidate, buffer, error) ==
            GPUXTB_STATUS_INVALID_ARGUMENT);
      CHECK(plan_unchanged());
      CHECK(attempt_hamiltonian_control_alias(fixture, candidate, buffer, error) ==
            GPUXTB_STATUS_INVALID_ARGUMENT);
      CHECK(plan_unchanged());
    }
  }

  for (std::size_t control = 0; control < 4u; ++control) {
    for (const std::size_t offset : {std::size_t{0}, sizeof(double)}) {
      for (std::size_t buffer = 0; buffer < 9u; ++buffer) {
        CHECK(attempt_population_descriptor_alias(fixture, control, offset, buffer, error) ==
              GPUXTB_STATUS_INVALID_ARGUMENT);
        CHECK(plan_unchanged());
        CHECK(attempt_hamiltonian_descriptor_alias(fixture, control, offset, buffer, error) ==
              GPUXTB_STATUS_INVALID_ARGUMENT);
        CHECK(plan_unchanged());
      }
    }
  }
  return 0;
}

int test_malformed_layout_alignment_and_partial_alias() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 2}, {1, 1}, {0.0}, {0}, {1}, fixture, error));
  const auto* const identity = fixture.plan.identity();

  IntegralPlan malformed_integrals = fixture.integral_plan;
  malformed_integrals.total_matrix_elements += 1;
  CHECK(gpuxtb::detail::gfn2::make_mulliken_plan(fixture.basis, malformed_integrals,
                                                 fixture.wavefunction, fixture.plan,
                                                 error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.plan.identity() == identity);

  WavefunctionLayout malformed_wavefunction = fixture.wavefunction;
  malformed_wavefunction.quadrupole.system_offsets.back() -= 1;
  CHECK(gpuxtb::detail::gfn2::make_mulliken_plan(fixture.basis, fixture.integral_plan,
                                                 malformed_wavefunction, fixture.plan,
                                                 error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.plan.identity() == identity);

  std::fill(fixture.overlap.begin(), fixture.overlap.end(), 0.2);
  std::fill(fixture.dipole_integrals.begin(), fixture.dipole_integrals.end(), 0.1);
  std::fill(fixture.quadrupole_integrals.begin(), fixture.quadrupole_integrals.end(), 0.05);
  std::fill(fixture.density.begin(), fixture.density.end(), 0.3);
  std::fill(fixture.qsh.begin(), fixture.qsh.end(), 7.0);
  std::vector<std::byte> misaligned_storage(fixture.density.size() * sizeof(double) + 1u);
  MullikenDensityView misaligned_density{
      reinterpret_cast<const double*>(misaligned_storage.data() + 1u),
      fixture.plan.density_elements()};
  CHECK(reinterpret_cast<std::uintptr_t>(misaligned_density.density) % alignof(double) != 0u);
  CHECK(gpuxtb::detail::gfn2::evaluate_mulliken_population_cpu(
            fixture.plan, integral_view(fixture), misaligned_density, population_view(fixture),
            workspace_view(fixture), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(fixture.qsh.begin(), fixture.qsh.end(),
                    [](double value) { return value == 7.0; }));

  std::vector<double> partial_alias_storage(5u, 9.0);
  MullikenPopulationView partial_alias = population_view(fixture);
  partial_alias.qsh = partial_alias_storage.data();
  partial_alias.qat = partial_alias_storage.data() + 1u;
  CHECK(gpuxtb::detail::gfn2::evaluate_mulliken_population_cpu(
            fixture.plan, integral_view(fixture), density_view(fixture), partial_alias,
            workspace_view(fixture), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(partial_alias_storage.begin(), partial_alias_storage.end(),
                    [](double value) { return value == 9.0; }));

  const std::size_t misaligned_elements = static_cast<std::size_t>(std::max(
      {fixture.plan.hamiltonian_scratch_elements(), fixture.plan.quadrupole_population_elements(),
       6 * fixture.plan.matrix_elements()}));
  std::vector<std::byte> large_misaligned_storage(misaligned_elements * sizeof(double) + 1u);
  auto* const misaligned = reinterpret_cast<double*>(large_misaligned_storage.data() + 1u);
  CHECK(reinterpret_cast<std::uintptr_t>(misaligned) % alignof(double) != 0u);

  const auto population_invalid =
      [&](const MullikenIntegralView& integrals, const MullikenDensityView& density,
          const MullikenPopulationView& population, const MullikenWorkspace& workspace) {
        std::fill(fixture.qsh.begin(), fixture.qsh.end(), 111.0);
        std::fill(fixture.qat.begin(), fixture.qat.end(), 112.0);
        std::fill(fixture.dipole.begin(), fixture.dipole.end(), 113.0);
        std::fill(fixture.quadrupole.begin(), fixture.quadrupole.end(), 114.0);
        const gpuxtb_status_t status = gpuxtb::detail::gfn2::evaluate_mulliken_population_cpu(
            fixture.plan, integrals, density, population, workspace, error);
        return status == GPUXTB_STATUS_INVALID_ARGUMENT && all_equal_to(fixture.qsh, 111.0) &&
               all_equal_to(fixture.qat, 112.0) && all_equal_to(fixture.dipole, 113.0) &&
               all_equal_to(fixture.quadrupole, 114.0);
      };
  const auto hamiltonian_invalid =
      [&](const MullikenIntegralView& integrals, const MullikenPotentialView& potential,
          const MullikenHamiltonianView& hamiltonian, const MullikenWorkspace& workspace) {
        std::fill(fixture.hamiltonian.begin(), fixture.hamiltonian.end(), 121.0);
        const gpuxtb_status_t status = gpuxtb::detail::gfn2::add_mulliken_hamiltonian_cpu(
            fixture.plan, integrals, potential, hamiltonian, workspace, error);
        return status == GPUXTB_STATUS_INVALID_ARGUMENT && all_equal_to(fixture.hamiltonian, 121.0);
      };

  {
    MullikenIntegralView view = integral_view(fixture);
    --view.matrix_elements;
    CHECK(population_invalid(view, density_view(fixture), population_view(fixture),
                             workspace_view(fixture)));
    CHECK(hamiltonian_invalid(view, potential_view(fixture), hamiltonian_view(fixture),
                              workspace_view(fixture)));
  }
  {
    MullikenDensityView view = density_view(fixture);
    --view.elements;
    CHECK(population_invalid(integral_view(fixture), view, population_view(fixture),
                             workspace_view(fixture)));
  }
  for (std::size_t count = 0; count < 4u; ++count) {
    MullikenPopulationView view = population_view(fixture);
    if (count == 0u) {
      --view.qsh_elements;
    } else if (count == 1u) {
      --view.qat_elements;
    } else if (count == 2u) {
      --view.dipole_elements;
    } else {
      --view.quadrupole_elements;
    }
    CHECK(population_invalid(integral_view(fixture), density_view(fixture), view,
                             workspace_view(fixture)));
  }
  for (std::size_t count = 0; count < 4u; ++count) {
    MullikenPotentialView view = potential_view(fixture);
    if (count == 0u) {
      --view.vat_elements;
    } else if (count == 1u) {
      --view.vsh_elements;
    } else if (count == 2u) {
      --view.dipole_elements;
    } else {
      --view.quadrupole_elements;
    }
    CHECK(hamiltonian_invalid(integral_view(fixture), view, hamiltonian_view(fixture),
                              workspace_view(fixture)));
  }
  {
    MullikenHamiltonianView view = hamiltonian_view(fixture);
    --view.elements;
    CHECK(hamiltonian_invalid(integral_view(fixture), potential_view(fixture), view,
                              workspace_view(fixture)));
  }
  {
    MullikenWorkspace view = workspace_view(fixture);
    view.elements = fixture.plan.population_scratch_elements() - 1;
    CHECK(population_invalid(integral_view(fixture), density_view(fixture),
                             population_view(fixture), view));
    view.elements = fixture.plan.hamiltonian_scratch_elements() - 1;
    CHECK(hamiltonian_invalid(integral_view(fixture), potential_view(fixture),
                              hamiltonian_view(fixture), view));
  }

  for (std::size_t pointer = 0; pointer < 3u; ++pointer) {
    MullikenIntegralView view = integral_view(fixture);
    if (pointer == 0u) {
      view.overlap = misaligned;
    } else if (pointer == 1u) {
      view.dipole = misaligned;
    } else {
      view.quadrupole = misaligned;
    }
    CHECK(population_invalid(view, density_view(fixture), population_view(fixture),
                             workspace_view(fixture)));
    CHECK(hamiltonian_invalid(view, potential_view(fixture), hamiltonian_view(fixture),
                              workspace_view(fixture)));
  }
  {
    MullikenDensityView view = density_view(fixture);
    view.density = misaligned;
    CHECK(population_invalid(integral_view(fixture), view, population_view(fixture),
                             workspace_view(fixture)));
  }
  for (std::size_t pointer = 0; pointer < 4u; ++pointer) {
    MullikenPopulationView view = population_view(fixture);
    if (pointer == 0u) {
      view.qsh = misaligned;
    } else if (pointer == 1u) {
      view.qat = misaligned;
    } else if (pointer == 2u) {
      view.dipole = misaligned;
    } else {
      view.quadrupole = misaligned;
    }
    CHECK(population_invalid(integral_view(fixture), density_view(fixture), view,
                             workspace_view(fixture)));
  }
  for (std::size_t pointer = 0; pointer < 4u; ++pointer) {
    MullikenPotentialView view = potential_view(fixture);
    if (pointer == 0u) {
      view.vat = misaligned;
    } else if (pointer == 1u) {
      view.vsh = misaligned;
    } else if (pointer == 2u) {
      view.dipole = misaligned;
    } else {
      view.quadrupole = misaligned;
    }
    CHECK(hamiltonian_invalid(integral_view(fixture), view, hamiltonian_view(fixture),
                              workspace_view(fixture)));
  }
  {
    MullikenHamiltonianView view = hamiltonian_view(fixture);
    view.matrix = misaligned;
    CHECK(hamiltonian_invalid(integral_view(fixture), potential_view(fixture), view,
                              workspace_view(fixture)));
  }
  {
    MullikenWorkspace view = workspace_view(fixture);
    view.scratch = misaligned;
    CHECK(population_invalid(integral_view(fixture), density_view(fixture),
                             population_view(fixture), view));
    CHECK(hamiltonian_invalid(integral_view(fixture), potential_view(fixture),
                              hamiltonian_view(fixture), view));
  }
  return 0;
}

}  // namespace

int main() {
  if (const int line = test_two_ao_population_fixture(); line != 0) {
    return line;
  }
  if (const int line = test_multi_ao_shell_ownership(); line != 0) {
    return line;
  }
  if (const int line = test_hamiltonian_two_directions_and_packed_dual(); line != 0) {
    return line;
  }
  if (const int line = test_spin_swap(); line != 0) {
    return line;
  }
  if (const int line = test_combined_adjoint_identity(); line != 0) {
    return line;
  }
  if (const int line = test_exact_plan_relationship_and_view_identity(); line != 0) {
    return line;
  }
  if (const int line = test_pinned_tblite_open_shell_literal_oracle(); line != 0) {
    return line;
  }
  if (const int line = test_open_shell_adjoint_identity(); line != 0) {
    return line;
  }
  if (const int line = test_batch_matches_sequential(); line != 0) {
    return line;
  }
  if (const int line = test_mixed_spin_heterogeneous_ragged_batch(); line != 0) {
    return line;
  }
  if (const int line = test_translation_invariance_with_integral_evaluator(); line != 0) {
    return line;
  }
  if (const int line = test_alias_atomicity_and_zero_allocation(); line != 0) {
    return line;
  }
  if (const int line = test_control_storage_alias_matrix(); line != 0) {
    return line;
  }
  if (const int line = test_malformed_layout_alignment_and_partial_alias(); line != 0) {
    return line;
  }
  return 0;
}
