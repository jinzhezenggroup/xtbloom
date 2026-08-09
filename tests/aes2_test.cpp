#include "model/gfn2/aes2.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "model/gfn2/coordination.hpp"

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

using xtbloom::detail::gfn2::AES2GeometryCache;
using xtbloom::detail::gfn2::AES2Plan;
using xtbloom::detail::gfn2::AES2Workspace;
using xtbloom::detail::gfn2::BasisPlan;
using xtbloom::detail::gfn2::CoordinationPlan;

static_assert(std::is_trivially_copyable_v<AES2GeometryCache>);
static_assert(std::is_standard_layout_v<AES2GeometryCache>);
static_assert(std::is_trivially_copyable_v<AES2Workspace>);
static_assert(std::is_standard_layout_v<AES2Workspace>);
static_assert(std::is_nothrow_copy_constructible_v<AES2Plan>);
static_assert(sizeof(AES2Plan) <= 4u * sizeof(void*));

constexpr std::uint64_t kGeneration = 37u;

struct Fixture {
  BasisPlan basis;
  CoordinationPlan coordination_plan;
  AES2Plan plan;
  std::vector<double> coordination_numbers;
  std::vector<double> pair_data;
  std::vector<double> pair_scratch;
  std::vector<double> potential_scratch;
  std::vector<double> batch_scratch;
  std::vector<double> gradient_scratch;
  std::vector<double> coordination_scratch;
  AES2Workspace workspace;
  AES2GeometryCache cache;
};

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

bool same_cache(const AES2GeometryCache& actual, const AES2GeometryCache& expected) {
  return actual.pair_data == expected.pair_data &&
         actual.pair_data_elements == expected.pair_data_elements &&
         actual.geometry_generation == expected.geometry_generation &&
         actual.plan_identity == expected.plan_identity;
}

bool make_fixture(const std::vector<std::int64_t>& offsets,
                  const std::vector<std::int32_t>& atomic_numbers,
                  const std::vector<double>& positions, Fixture& fixture, std::string& error) {
  const std::int64_t batch_size = static_cast<std::int64_t>(offsets.size() - 1u);
  const std::int64_t atom_count = static_cast<std::int64_t>(atomic_numbers.size());
  if (xtbloom::detail::gfn2::make_basis_plan(batch_size, atom_count, offsets.data(),
                                             atomic_numbers.data(), fixture.basis,
                                             error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_coordination_plan(
          batch_size, atom_count, offsets.data(), atomic_numbers.data(), fixture.coordination_plan,
          error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_aes2_plan(fixture.basis, atomic_numbers.data(), fixture.plan,
                                            error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }

  fixture.coordination_numbers.resize(static_cast<std::size_t>(atom_count));
  if (xtbloom::detail::gfn2::evaluate_coordination_cpu(fixture.coordination_plan, positions.data(),
                                                       fixture.coordination_numbers.data(),
                                                       error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  fixture.pair_data.resize(static_cast<std::size_t>(fixture.plan.pair_data_elements()));
  fixture.pair_scratch.resize(static_cast<std::size_t>(fixture.plan.pair_data_elements()));
  fixture.potential_scratch.resize(
      static_cast<std::size_t>(fixture.plan.potential_scratch_elements()));
  fixture.batch_scratch.resize(static_cast<std::size_t>(fixture.plan.batch_size()));
  fixture.gradient_scratch.resize(
      static_cast<std::size_t>(fixture.plan.gradient_scratch_elements()));
  fixture.coordination_scratch.resize(
      static_cast<std::size_t>(fixture.plan.coordination_scratch_elements()));
  fixture.workspace = AES2Workspace{
      fixture.pair_scratch.data(),         fixture.plan.pair_data_elements(),
      fixture.potential_scratch.data(),    fixture.plan.potential_scratch_elements(),
      fixture.batch_scratch.data(),        fixture.plan.batch_size(),
      fixture.gradient_scratch.data(),     fixture.plan.gradient_scratch_elements(),
      fixture.coordination_scratch.data(), fixture.plan.coordination_scratch_elements(),
  };
  return xtbloom::detail::gfn2::update_aes2_geometry_cache_cpu(
             fixture.plan, positions.data(), fixture.coordination_numbers.data(), kGeneration,
             fixture.pair_data.data(), fixture.pair_data.size(), fixture.workspace, fixture.cache,
             error) == XTBLOOM_STATUS_SUCCESS;
}

bool energy(const Fixture& fixture, const std::vector<double>& charges,
            const std::vector<double>& dipoles, const std::vector<double>& quadrupoles,
            std::vector<double>& energies, std::string& error) {
  return xtbloom::detail::gfn2::add_aes2_energy_cpu(
             fixture.plan, fixture.cache, charges.data(), dipoles.data(), quadrupoles.data(),
             energies.data(), fixture.workspace, error) == XTBLOOM_STATUS_SUCCESS;
}

bool update_geometry(Fixture& fixture, const std::vector<double>& positions,
                     const std::vector<double>& coordination_numbers,
                     std::uint64_t geometry_generation, std::string& error) {
  return xtbloom::detail::gfn2::update_aes2_geometry_cache_cpu(
             fixture.plan, positions.data(), coordination_numbers.data(), geometry_generation,
             fixture.pair_data.data(), fixture.pair_data.size(), fixture.workspace, fixture.cache,
             error) == XTBLOOM_STATUS_SUCCESS;
}

bool vjp(const Fixture& fixture, const std::vector<double>& positions,
         const std::vector<double>& coordination_numbers, std::uint64_t geometry_generation,
         const std::vector<double>& charges, const std::vector<double>& dipoles,
         const std::vector<double>& quadrupoles, std::vector<double>& gradients,
         std::vector<double>& coordination_adjoints, std::string& error) {
  return xtbloom::detail::gfn2::add_aes2_vjp_cpu(
             fixture.plan, fixture.cache, positions.data(), coordination_numbers.data(),
             geometry_generation, charges.data(), dipoles.data(), quadrupoles.data(),
             gradients.data(), coordination_adjoints.data(), fixture.workspace,
             error) == XTBLOOM_STATUS_SUCCESS;
}

int test_dxtb_oracles_and_isolated_components() {
  const std::vector<std::int64_t> offsets{0, 2};
  const std::vector<std::int32_t> numbers{3, 1};
  const std::vector<double> positions{
      0.0, 0.0, -1.50796743897235, 0.0, 0.0, 1.50796743897235,
  };
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(offsets, numbers, positions, fixture, error));
  CHECK(error.empty());
  CHECK(fixture.plan.total_pairs() == 1);
  CHECK(fixture.plan.pair_data_elements() == 5);
  CHECK(near(fixture.coordination_numbers[0], 0.9369498285810679, 2.0e-15));
  CHECK(fixture.coordination_numbers[0] == fixture.coordination_numbers[1]);
  CHECK(fixture.pair_data[0] == 0.0);
  CHECK(fixture.pair_data[1] == 0.0);
  CHECK(fixture.pair_data[2] == -3.0159348779447);
  CHECK(near(fixture.pair_data[3], 0.0044217191381834085, 3.0e-18));
  CHECK(near(fixture.pair_data[4], 0.00045992656077582897, 5.0e-19));

  /*
   * The multipoles and fixed results below come from dxtb's Apache-2.0
   * test/test_coulomb/test_aes2.py. They in turn exercise the same tblite
   * get_mrad/get_multipole_matrix_0d/get_energy implementation pinned by
   * xtbloom's generated GFN2 parameter table.
   */
  const std::vector<double> charges{0.54699448343345114, -0.54699448343345114};
  const std::vector<double> dipoles{
      0.0, 0.0, -1.1260506806881299, 0.0, 0.0, 0.079884324667409912,
  };
  const std::vector<double> quadrupoles{
      1.4096150819303312,     0.0, 1.4096150819303312,     0.0, 0.0, -2.8192301638606625,
      -0.0023636549459148497, 0.0, -0.0023636549459148497, 0.0, 0.0, 0.004727309891829741,
  };
  const std::vector<double> zero_charges(2, 0.0);
  const std::vector<double> zero_dipoles(6, 0.0);
  const std::vector<double> zero_quadrupoles(12, 0.0);

  std::vector<double> energies(1, 0.0);
  CHECK(energy(fixture, charges, dipoles, zero_quadrupoles, energies, error));
  const double dipole_onsite = -0.005984890804436024;
  const double charge_dipole = 0.007631269810985883;
  const double dipole_dipole = 0.0007526306690281942;
  CHECK(near(energies[0], dipole_onsite + charge_dipole + dipole_dipole, 8.0e-18));

  energies[0] = 0.0;
  CHECK(energy(fixture, charges, zero_dipoles, quadrupoles, energies, error));
  const double quadrupole_onsite = 0.0023844268102436858;
  const double charge_quadrupole = 0.0064620975395246584;
  CHECK(near(energies[0], quadrupole_onsite + charge_quadrupole, 8.0e-18));

  energies[0] = 0.0;
  CHECK(energy(fixture, zero_charges, dipoles, zero_quadrupoles, energies, error));
  CHECK(near(energies[0], dipole_onsite + dipole_dipole, 6.0e-18));
  energies[0] = 0.0;
  CHECK(energy(fixture, zero_charges, zero_dipoles, quadrupoles, energies, error));
  CHECK(near(energies[0], quadrupole_onsite, 4.0e-18));
  energies[0] = 0.0;
  CHECK(energy(fixture, charges, dipoles, quadrupoles, energies, error));
  CHECK(near(energies[0], 0.011245534025346397, 1.0e-17));

  const std::vector<double> potential_charges{0.44132071106699799, -0.44132071106699783};
  const std::vector<double> potential_dipoles{
      0.0, 0.0, -0.45042027227525194, 0.0, 0.0, 0.031953729866963966,
  };
  const std::vector<double> potential_quadrupoles{
      0.56384603277213252,     0.0, 0.56384603277213252,     0.0, 0.0, -1.127692065544265,
      -0.00094546197836593989, 0.0, -0.00094546197836593989, 0.0, 0.0, 0.0018909239567318965,
  };
  std::vector<double> charge_potential(2);
  std::vector<double> dipole_potential(6);
  std::vector<double> quadrupole_potential(12);
  CHECK(xtbloom::detail::gfn2::evaluate_aes2_potential_cpu(
            fixture.plan, fixture.cache, potential_charges.data(), potential_dipoles.data(),
            potential_quadrupoles.data(), charge_potential.data(), dipole_potential.data(),
            quadrupole_potential.data(), fixture.workspace, error) == XTBLOOM_STATUS_SUCCESS);
  constexpr std::array<double, 2> expected_charge{
      -0.00041821215599096711,
      -0.010724251999060196,
  };
  constexpr std::array<double, 6> expected_dipole{
      0.0, 0.0, -0.0016484335657207883, 0.0, 0.0, 0.0014390586834179975,
  };
  constexpr std::array<double, 12> expected_quadrupole{
      0.00022553841310885301,     0.0, 0.00022553841310885301,     0.0, 0.0, -0.0022973107195650411,
      -0.00000051869935057112189, 0.0, -0.00000051869935057112189, 0.0, 0.0, 0.001847271292048478,
  };
  for (std::size_t atom = 0; atom < expected_charge.size(); ++atom) {
    CHECK(near(charge_potential[atom], expected_charge[atom], 4.0e-18));
  }
  for (std::size_t component = 0; component < expected_dipole.size(); ++component) {
    CHECK(near(dipole_potential[component], expected_dipole[component], 4.0e-18));
  }
  for (std::size_t component = 0; component < expected_quadrupole.size(); ++component) {
    CHECK(near(quadrupole_potential[component], expected_quadrupole[component], 4.0e-18));
  }
  return 0;
}

int test_coordinate_and_cn_vjp_finite_differences() {
  const std::vector<std::int64_t> offsets{0, 4};
  const std::vector<std::int32_t> numbers{6, 8, 14, 1};
  std::vector<double> positions{
      -0.7, 0.2, 0.8, 1.1, -0.9, 0.3, 0.4, 1.8, -0.6, 2.5, 0.7, 1.2,
  };
  std::vector<double> coordination_numbers{1.15, 2.2, 0.75, 1.6};
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(offsets, numbers, positions, fixture, error));

  const std::vector<double> charges{0.21, -0.34, 0.08, 0.05};
  const std::vector<double> dipoles{
      0.13, -0.08, 0.04, -0.11, 0.06, 0.09, 0.03, 0.12, -0.05, -0.07, 0.02, 0.1,
  };
  const std::vector<double> quadrupoles{
      0.12, 0.03,  -0.07, -0.02, 0.04, -0.05, -0.09, 0.05,  0.02,  0.07, -0.03, 0.07,
      0.03, -0.04, -0.08, 0.01,  0.02, 0.05,  0.08,  -0.02, -0.01, 0.06, 0.03,  -0.07,
  };
  const std::vector<double> zero_charges(numbers.size(), 0.0);
  const std::vector<double> zero_dipoles(numbers.size() * 3u, 0.0);
  const std::vector<double> zero_quadrupoles(numbers.size() * 6u, 0.0);
  std::vector<double> charge_dipole_moments(zero_dipoles);
  std::copy_n(dipoles.begin(), 3u, charge_dipole_moments.begin());
  std::vector<double> charge_quadrupole_moments(zero_quadrupoles);
  std::copy_n(quadrupoles.begin() + 12, 6u, charge_quadrupole_moments.begin() + 12);

  struct InteractionCase {
    const std::vector<double>* charges;
    const std::vector<double>* dipoles;
    const std::vector<double>* quadrupoles;
  };
  const std::array<InteractionCase, 4> cases{{
      {&charges, &charge_dipole_moments, &zero_quadrupoles},
      {&zero_charges, &dipoles, &zero_quadrupoles},
      {&charges, &zero_dipoles, &charge_quadrupole_moments},
      {&charges, &dipoles, &quadrupoles},
  }};

  constexpr double coordinate_step = 2.0e-5;
  constexpr double cn_step = 2.0e-6;
  for (std::size_t interaction = 0; interaction < cases.size(); ++interaction) {
    const std::uint64_t generation = 800u + static_cast<std::uint64_t>(interaction);
    CHECK(update_geometry(fixture, positions, coordination_numbers, generation, error));
    std::vector<double> gradient_seed(positions.size());
    std::vector<double> cn_seed(coordination_numbers.size());
    for (std::size_t coordinate = 0; coordinate < gradient_seed.size(); ++coordinate) {
      gradient_seed[coordinate] = 0.003 * static_cast<double>(coordinate + 1u);
    }
    for (std::size_t atom = 0; atom < cn_seed.size(); ++atom) {
      cn_seed[atom] = -0.007 * static_cast<double>(atom + 1u);
    }
    std::vector<double> gradients = gradient_seed;
    std::vector<double> coordination_adjoints = cn_seed;
    CHECK(vjp(fixture, positions, coordination_numbers, generation, *cases[interaction].charges,
              *cases[interaction].dipoles, *cases[interaction].quadrupoles, gradients,
              coordination_adjoints, error));

    for (std::size_t coordinate = 0; coordinate < positions.size(); ++coordinate) {
      positions[coordinate] += coordinate_step;
      CHECK(update_geometry(fixture, positions, coordination_numbers, generation + 100u, error));
      std::vector<double> right(1, 0.0);
      CHECK(energy(fixture, *cases[interaction].charges, *cases[interaction].dipoles,
                   *cases[interaction].quadrupoles, right, error));
      positions[coordinate] -= 2.0 * coordinate_step;
      CHECK(update_geometry(fixture, positions, coordination_numbers, generation + 101u, error));
      std::vector<double> left(1, 0.0);
      CHECK(energy(fixture, *cases[interaction].charges, *cases[interaction].dipoles,
                   *cases[interaction].quadrupoles, left, error));
      positions[coordinate] += coordinate_step;
      CHECK(near((right[0] - left[0]) / (2.0 * coordinate_step),
                 gradients[coordinate] - gradient_seed[coordinate], 8.0e-10));
    }

    for (std::size_t atom = 0; atom < coordination_numbers.size(); ++atom) {
      coordination_numbers[atom] += cn_step;
      CHECK(update_geometry(fixture, positions, coordination_numbers, generation + 200u, error));
      std::vector<double> right(1, 0.0);
      CHECK(energy(fixture, *cases[interaction].charges, *cases[interaction].dipoles,
                   *cases[interaction].quadrupoles, right, error));
      coordination_numbers[atom] -= 2.0 * cn_step;
      CHECK(update_geometry(fixture, positions, coordination_numbers, generation + 201u, error));
      std::vector<double> left(1, 0.0);
      CHECK(energy(fixture, *cases[interaction].charges, *cases[interaction].dipoles,
                   *cases[interaction].quadrupoles, left, error));
      coordination_numbers[atom] += cn_step;
      CHECK(near((right[0] - left[0]) / (2.0 * cn_step),
                 coordination_adjoints[atom] - cn_seed[atom], 2.0e-10));
    }

    for (std::size_t axis = 0; axis < 3u; ++axis) {
      double sum = 0.0;
      for (std::size_t atom = 0; atom < numbers.size(); ++atom) {
        sum += gradients[atom * 3u + axis] - gradient_seed[atom * 3u + axis];
      }
      CHECK(near(sum, 0.0, 4.0e-17));
    }

    std::vector<double> translated_positions = positions;
    for (std::size_t atom = 0; atom < numbers.size(); ++atom) {
      translated_positions[atom * 3u] += 4.5;
      translated_positions[atom * 3u + 1u] -= 2.75;
      translated_positions[atom * 3u + 2u] += 1.125;
    }
    CHECK(update_geometry(fixture, translated_positions, coordination_numbers, generation + 300u,
                          error));
    std::vector<double> translated_gradients = gradient_seed;
    std::vector<double> translated_cn = cn_seed;
    CHECK(vjp(fixture, translated_positions, coordination_numbers, generation + 300u,
              *cases[interaction].charges, *cases[interaction].dipoles,
              *cases[interaction].quadrupoles, translated_gradients, translated_cn, error));
    for (std::size_t coordinate = 0; coordinate < gradients.size(); ++coordinate) {
      CHECK(near(translated_gradients[coordinate], gradients[coordinate], 5.0e-16));
    }
    for (std::size_t atom = 0; atom < coordination_adjoints.size(); ++atom) {
      CHECK(near(translated_cn[atom], coordination_adjoints[atom], 3.0e-16));
    }
  }

  /* Onsite d/Q kernels are element constants, hence their explicit VJP is zero. */
  const std::vector<std::int64_t> isolated_offsets{0, 1, 2, 3, 4};
  Fixture isolated;
  CHECK(make_fixture(isolated_offsets, numbers, positions, isolated, error));
  CHECK(update_geometry(isolated, positions, coordination_numbers, 1200u, error));
  std::vector<double> onsite_gradients(positions.size(), 0.125);
  std::vector<double> onsite_cn(coordination_numbers.size(), -0.25);
  CHECK(vjp(isolated, positions, coordination_numbers, 1200u, zero_charges, dipoles, quadrupoles,
            onsite_gradients, onsite_cn, error));
  CHECK(std::all_of(onsite_gradients.begin(), onsite_gradients.end(),
                    [](double value) { return value == 0.125; }));
  CHECK(
      std::all_of(onsite_cn.begin(), onsite_cn.end(), [](double value) { return value == -0.25; }));
  std::vector<double> onsite_energy(4, 0.0);
  CHECK(energy(isolated, zero_charges, dipoles, quadrupoles, onsite_energy, error));
  CHECK(std::any_of(onsite_energy.begin(), onsite_energy.end(),
                    [](double value) { return value != 0.0; }));

  /* Compose the CN adjoint with GFN2 coordination to recover the total dE/dR. */
  Fixture composed;
  CHECK(make_fixture(offsets, numbers, positions, composed, error));
  std::vector<double> composed_gradient(positions.size(), 0.0);
  std::vector<double> composed_cn(numbers.size(), 0.0);
  CHECK(vjp(composed, positions, composed.coordination_numbers, kGeneration, charges, dipoles,
            quadrupoles, composed_gradient, composed_cn, error));
  CHECK(xtbloom::detail::gfn2::add_coordination_gradient_cpu(
            composed.coordination_plan, positions.data(), composed_cn.data(),
            composed_gradient.data(), error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t coordinate = 0; coordinate < positions.size(); ++coordinate) {
    positions[coordinate] += coordinate_step;
    std::vector<double> right_cn(numbers.size());
    CHECK(xtbloom::detail::gfn2::evaluate_coordination_cpu(composed.coordination_plan,
                                                           positions.data(), right_cn.data(),
                                                           error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(update_geometry(composed, positions, right_cn, 1400u, error));
    std::vector<double> right(1, 0.0);
    CHECK(energy(composed, charges, dipoles, quadrupoles, right, error));
    positions[coordinate] -= 2.0 * coordinate_step;
    std::vector<double> left_cn(numbers.size());
    CHECK(xtbloom::detail::gfn2::evaluate_coordination_cpu(composed.coordination_plan,
                                                           positions.data(), left_cn.data(),
                                                           error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(update_geometry(composed, positions, left_cn, 1401u, error));
    std::vector<double> left(1, 0.0);
    CHECK(energy(composed, charges, dipoles, quadrupoles, left, error));
    positions[coordinate] += coordinate_step;
    CHECK(near((right[0] - left[0]) / (2.0 * coordinate_step), composed_gradient[coordinate],
               1.5e-9));
  }
  return 0;
}

int test_energy_potential_consistency() {
  const std::vector<std::int64_t> offsets{0, 4};
  const std::vector<std::int32_t> numbers{6, 8, 14, 1};
  const std::vector<double> positions{
      -0.7, 0.2, 0.8, 1.1, -0.9, 0.3, 0.4, 1.8, -0.6, 2.5, 0.7, 1.2,
  };
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(offsets, numbers, positions, fixture, error));
  std::vector<double> charges(numbers.size());
  std::vector<double> dipoles(numbers.size() * 3u);
  std::vector<double> quadrupoles(numbers.size() * 6u);
  for (std::size_t atom = 0; atom < numbers.size(); ++atom) {
    charges[atom] = 0.17 * std::sin(0.8 * static_cast<double>(atom + 1u)) - 0.04;
    for (std::size_t component = 0; component < 3u; ++component) {
      dipoles[atom * 3u + component] =
          0.11 * std::cos(0.47 * static_cast<double>(atom * 3u + component + 2u));
    }
    const double xx = 0.09 * std::sin(0.31 * static_cast<double>(atom + 1u));
    const double yy = -0.06 * std::cos(0.53 * static_cast<double>(atom + 2u));
    quadrupoles[atom * 6u] = xx;
    quadrupoles[atom * 6u + 1u] = 0.03 * static_cast<double>(atom + 1u);
    quadrupoles[atom * 6u + 2u] = yy;
    quadrupoles[atom * 6u + 3u] = -0.02 * static_cast<double>(atom + 2u);
    quadrupoles[atom * 6u + 4u] = 0.015 * static_cast<double>(atom + 3u);
    quadrupoles[atom * 6u + 5u] = -xx - yy;
  }

  std::vector<double> charge_potential(numbers.size());
  std::vector<double> dipole_potential(dipoles.size());
  std::vector<double> quadrupole_potential(quadrupoles.size());
  CHECK(xtbloom::detail::gfn2::evaluate_aes2_potential_cpu(
            fixture.plan, fixture.cache, charges.data(), dipoles.data(), quadrupoles.data(),
            charge_potential.data(), dipole_potential.data(), quadrupole_potential.data(),
            fixture.workspace, error) == XTBLOOM_STATUS_SUCCESS);
  std::vector<double> energies(1, 0.0);
  CHECK(energy(fixture, charges, dipoles, quadrupoles, energies, error));
  double contraction = 0.0;
  for (std::size_t atom = 0; atom < charges.size(); ++atom) {
    contraction += charges[atom] * charge_potential[atom];
  }
  for (std::size_t component = 0; component < dipoles.size(); ++component) {
    contraction += dipoles[component] * dipole_potential[component];
  }
  for (std::size_t component = 0; component < quadrupoles.size(); ++component) {
    contraction += quadrupoles[component] * quadrupole_potential[component];
  }
  CHECK(near(energies[0], 0.5 * contraction, 2.0e-16));

  constexpr double step = 1.0e-6;
  const auto finite_difference = [&](std::vector<double>& variables, std::size_t index,
                                     double expected) {
    variables[index] += step;
    std::vector<double> right(1, 0.0);
    const bool right_ok = energy(fixture, charges, dipoles, quadrupoles, right, error);
    variables[index] -= 2.0 * step;
    std::vector<double> left(1, 0.0);
    const bool left_ok = energy(fixture, charges, dipoles, quadrupoles, left, error);
    variables[index] += step;
    return right_ok && left_ok && near((right[0] - left[0]) / (2.0 * step), expected, 2.0e-11);
  };
  for (std::size_t index = 0; index < charges.size(); ++index) {
    CHECK(finite_difference(charges, index, charge_potential[index]));
  }
  for (std::size_t index = 0; index < dipoles.size(); ++index) {
    CHECK(finite_difference(dipoles, index, dipole_potential[index]));
  }
  for (std::size_t index = 0; index < quadrupoles.size(); ++index) {
    CHECK(finite_difference(quadrupoles, index, quadrupole_potential[index]));
  }
  return 0;
}

using Matrix3 = std::array<double, 9>;

std::array<double, 3> rotate_vector(const Matrix3& rotation, const double* vector) {
  std::array<double, 3> result{};
  for (std::size_t row = 0; row < 3u; ++row) {
    for (std::size_t column = 0; column < 3u; ++column) {
      result[row] += rotation[row * 3u + column] * vector[column];
    }
  }
  return result;
}

std::array<double, 6> rotate_quadrupole(const Matrix3& rotation, const double* packed) {
  const Matrix3 tensor{{packed[0], packed[1], packed[3], packed[1], packed[2], packed[4], packed[3],
                        packed[4], packed[5]}};
  Matrix3 rotated{};
  for (std::size_t row = 0; row < 3u; ++row) {
    for (std::size_t column = 0; column < 3u; ++column) {
      for (std::size_t left = 0; left < 3u; ++left) {
        for (std::size_t right = 0; right < 3u; ++right) {
          rotated[row * 3u + column] +=
              rotation[row * 3u + left] * tensor[left * 3u + right] * rotation[column * 3u + right];
        }
      }
    }
  }
  return {{rotated[0], rotated[1], rotated[4], rotated[2], rotated[5], rotated[8]}};
}

std::array<double, 6> rotate_quadrupole_dual(const Matrix3& rotation, const double* packed) {
  const std::array<double, 6> physical{
      {packed[0], 0.5 * packed[1], packed[2], 0.5 * packed[3], 0.5 * packed[4], packed[5]}};
  const std::array<double, 6> rotated = rotate_quadrupole(rotation, physical.data());
  return {
      {rotated[0], 2.0 * rotated[1], rotated[2], 2.0 * rotated[3], 2.0 * rotated[4], rotated[5]}};
}

int test_rotation_covariance_and_tracelessness() {
  const std::vector<std::int64_t> offsets{0, 3};
  const std::vector<std::int32_t> numbers{8, 6, 1};
  const std::vector<double> positions{-0.8, 0.4, 0.2, 1.3, -0.7, 0.9, 0.5, 1.6, -1.1};
  const std::vector<double> charges{0.24, -0.31, 0.07};
  const std::vector<double> dipoles{0.13, -0.08, 0.04, -0.11, 0.06, 0.09, 0.03, 0.12, -0.05};
  const std::vector<double> quadrupoles{
      0.12, 0.03,  -0.07, -0.02, 0.04,  -0.05, -0.09, 0.05, 0.02,
      0.07, -0.03, 0.07,  0.03,  -0.04, -0.08, 0.01,  0.02, 0.05,
  };
  Fixture original;
  std::string error;
  CHECK(make_fixture(offsets, numbers, positions, original, error));
  const std::vector<double> coordination_numbers{1.3, 2.1, 0.8};
  CHECK(update_geometry(original, positions, coordination_numbers, 1300u, error));
  std::vector<double> original_charge_potential(3);
  std::vector<double> original_dipole_potential(9);
  std::vector<double> original_quadrupole_potential(18);
  CHECK(xtbloom::detail::gfn2::evaluate_aes2_potential_cpu(
            original.plan, original.cache, charges.data(), dipoles.data(), quadrupoles.data(),
            original_charge_potential.data(), original_dipole_potential.data(),
            original_quadrupole_potential.data(), original.workspace,
            error) == XTBLOOM_STATUS_SUCCESS);
  std::vector<double> original_energy(1, 0.0);
  CHECK(energy(original, charges, dipoles, quadrupoles, original_energy, error));
  std::vector<double> original_gradient(positions.size(), 0.0);
  std::vector<double> original_cn(numbers.size(), 0.0);
  CHECK(vjp(original, positions, coordination_numbers, 1300u, charges, dipoles, quadrupoles,
            original_gradient, original_cn, error));

  constexpr double cosine = 0.36;
  constexpr double sine = 0.9329523031752481;
  const Matrix3 rotation{{cosine, -sine, 0.0, 0.5 * sine, 0.5 * cosine, -0.8660254037844386,
                          0.8660254037844386 * sine, 0.8660254037844386 * cosine, 0.5}};
  std::vector<double> rotated_positions(positions.size());
  std::vector<double> rotated_dipoles(dipoles.size());
  std::vector<double> rotated_quadrupoles(quadrupoles.size());
  for (std::size_t atom = 0; atom < numbers.size(); ++atom) {
    const std::array<double, 3> position = rotate_vector(rotation, positions.data() + atom * 3u);
    const std::array<double, 3> dipole = rotate_vector(rotation, dipoles.data() + atom * 3u);
    const std::array<double, 6> quadrupole =
        rotate_quadrupole(rotation, quadrupoles.data() + atom * 6u);
    std::copy(position.begin(), position.end(), rotated_positions.data() + atom * 3u);
    std::copy(dipole.begin(), dipole.end(), rotated_dipoles.data() + atom * 3u);
    std::copy(quadrupole.begin(), quadrupole.end(), rotated_quadrupoles.data() + atom * 6u);
    CHECK(near(quadrupole[0] + quadrupole[2] + quadrupole[5], 0.0, 6.0e-17));
  }

  Fixture rotated;
  CHECK(make_fixture(offsets, numbers, rotated_positions, rotated, error));
  CHECK(update_geometry(rotated, rotated_positions, coordination_numbers, 1301u, error));
  std::vector<double> rotated_charge_potential(3);
  std::vector<double> rotated_dipole_potential(9);
  std::vector<double> rotated_quadrupole_potential(18);
  CHECK(xtbloom::detail::gfn2::evaluate_aes2_potential_cpu(
            rotated.plan, rotated.cache, charges.data(), rotated_dipoles.data(),
            rotated_quadrupoles.data(), rotated_charge_potential.data(),
            rotated_dipole_potential.data(), rotated_quadrupole_potential.data(), rotated.workspace,
            error) == XTBLOOM_STATUS_SUCCESS);
  std::vector<double> rotated_energy(1, 0.0);
  CHECK(energy(rotated, charges, rotated_dipoles, rotated_quadrupoles, rotated_energy, error));
  CHECK(near(rotated_energy[0], original_energy[0], 2.0e-16));
  std::vector<double> rotated_gradient(rotated_positions.size(), 0.0);
  std::vector<double> rotated_cn(numbers.size(), 0.0);
  CHECK(vjp(rotated, rotated_positions, coordination_numbers, 1301u, charges, rotated_dipoles,
            rotated_quadrupoles, rotated_gradient, rotated_cn, error));

  for (std::size_t atom = 0; atom < numbers.size(); ++atom) {
    CHECK(near(rotated_charge_potential[atom], original_charge_potential[atom], 2.0e-16));
    const std::array<double, 3> expected_dipole =
        rotate_vector(rotation, original_dipole_potential.data() + atom * 3u);
    const std::array<double, 6> expected_quadrupole =
        rotate_quadrupole_dual(rotation, original_quadrupole_potential.data() + atom * 6u);
    for (std::size_t component = 0; component < 3u; ++component) {
      CHECK(near(rotated_dipole_potential[atom * 3u + component], expected_dipole[component],
                 2.0e-16));
    }
    for (std::size_t component = 0; component < 6u; ++component) {
      CHECK(near(rotated_quadrupole_potential[atom * 6u + component],
                 expected_quadrupole[component], 3.0e-16));
    }
    const std::array<double, 3> expected_gradient =
        rotate_vector(rotation, original_gradient.data() + atom * 3u);
    for (std::size_t component = 0; component < 3u; ++component) {
      CHECK(near(rotated_gradient[atom * 3u + component], expected_gradient[component], 4.0e-16));
    }
    CHECK(near(rotated_cn[atom], original_cn[atom], 3.0e-16));
  }
  return 0;
}

int test_ragged_matches_sequential() {
  const std::vector<std::int64_t> offsets{0, 2, 2, 5, 6};
  const std::vector<std::int32_t> numbers{3, 1, 8, 6, 1, 79};
  const std::vector<double> positions{
      0.0, 0.0, -1.2, 0.1, 0.2, 1.4, -0.7, 0.5, 0.1, 1.2, -0.8, 0.9, 2.1, 0.4, -1.0, 3.3, -0.6, 0.7,
  };
  Fixture batch;
  std::string error;
  CHECK(make_fixture(offsets, numbers, positions, batch, error));
  std::vector<double> charges(numbers.size());
  std::vector<double> dipoles(numbers.size() * 3u);
  std::vector<double> quadrupoles(numbers.size() * 6u);
  for (std::size_t atom = 0; atom < numbers.size(); ++atom) {
    charges[atom] = 0.08 * static_cast<double>(atom + 1u) - 0.27;
    for (std::size_t component = 0; component < 3u; ++component) {
      dipoles[atom * 3u + component] =
          0.03 * static_cast<double>(atom * 3u + component + 1u) - 0.19;
    }
    const double xx = 0.02 * static_cast<double>(atom + 1u);
    const double yy = -0.015 * static_cast<double>(atom + 2u);
    quadrupoles[atom * 6u] = xx;
    quadrupoles[atom * 6u + 1u] = -0.01 * static_cast<double>(atom + 1u);
    quadrupoles[atom * 6u + 2u] = yy;
    quadrupoles[atom * 6u + 3u] = 0.012 * static_cast<double>(atom + 1u);
    quadrupoles[atom * 6u + 4u] = -0.008 * static_cast<double>(atom + 2u);
    quadrupoles[atom * 6u + 5u] = -xx - yy;
  }
  std::vector<double> batch_charge_potential(numbers.size());
  std::vector<double> batch_dipole_potential(dipoles.size());
  std::vector<double> batch_quadrupole_potential(quadrupoles.size());
  std::vector<double> batch_energy(4, 0.0);
  std::vector<double> batch_gradient(positions.size(), 0.0);
  std::vector<double> batch_cn(numbers.size(), 0.0);
  CHECK(xtbloom::detail::gfn2::evaluate_aes2_potential_cpu(
            batch.plan, batch.cache, charges.data(), dipoles.data(), quadrupoles.data(),
            batch_charge_potential.data(), batch_dipole_potential.data(),
            batch_quadrupole_potential.data(), batch.workspace, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(energy(batch, charges, dipoles, quadrupoles, batch_energy, error));
  CHECK(vjp(batch, positions, batch.coordination_numbers, kGeneration, charges, dipoles,
            quadrupoles, batch_gradient, batch_cn, error));
  CHECK(batch_energy[1] == 0.0);

  /* The system primitive must reproduce the batch result, including an empty
   * ragged member, while preserving accumulation semantics. */
  for (std::int64_t system = 0; system < batch.plan.batch_size(); ++system) {
    double system_energy = 0.375;
    CHECK(xtbloom::detail::gfn2::add_aes2_energy_system_cpu(
              batch.plan, batch.cache, system, charges.data(), dipoles.data(), quadrupoles.data(),
              system_energy, batch.workspace, error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(system_energy == 0.375 + batch_energy[static_cast<std::size_t>(system)]);
  }

  /* Numerical poison in a peer atom and peer cache slice is intentionally
   * invisible to the selected system. Selecting the poisoned member then
   * reports an isolated numerical failure without publishing its energy. */
  const std::size_t peer_atom = 2u;
  const std::size_t peer_pair = static_cast<std::size_t>(batch.plan.pair_offsets()[2]) * 5u;
  const double saved_charge = charges[peer_atom];
  const double saved_dipole = dipoles[peer_atom * 3u];
  const double saved_quadrupole = quadrupoles[peer_atom * 6u];
  const double saved_pair = batch.pair_data[peer_pair];
  charges[peer_atom] = std::numeric_limits<double>::quiet_NaN();
  dipoles[peer_atom * 3u] = std::numeric_limits<double>::infinity();
  quadrupoles[peer_atom * 6u] = std::numeric_limits<double>::quiet_NaN();
  batch.pair_data[peer_pair] = std::numeric_limits<double>::quiet_NaN();
  double isolated_energy = -0.25;
  CHECK(xtbloom::detail::gfn2::add_aes2_energy_system_cpu(
            batch.plan, batch.cache, 0, charges.data(), dipoles.data(), quadrupoles.data(),
            isolated_energy, batch.workspace, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(isolated_energy == -0.25 + batch_energy[0]);
  double failed_energy = 9.5;
  CHECK(xtbloom::detail::gfn2::add_aes2_energy_system_cpu(
            batch.plan, batch.cache, 2, charges.data(), dipoles.data(), quadrupoles.data(),
            failed_energy, batch.workspace, error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(failed_energy == 9.5);
  charges[peer_atom] = saved_charge;
  dipoles[peer_atom * 3u] = saved_dipole;
  quadrupoles[peer_atom * 6u] = saved_quadrupole;
  batch.pair_data[peer_pair] = saved_pair;

  for (std::size_t system : {0u, 2u, 3u}) {
    const std::int64_t begin = offsets[system];
    const std::int64_t end = offsets[system + 1u];
    const std::size_t atom_count = static_cast<std::size_t>(end - begin);
    const std::vector<std::int64_t> sequential_offsets{0, end - begin};
    const std::vector<std::int32_t> sequential_numbers(numbers.begin() + begin,
                                                       numbers.begin() + end);
    const std::vector<double> sequential_positions(positions.begin() + begin * 3,
                                                   positions.begin() + end * 3);
    const std::vector<double> sequential_charges(charges.begin() + begin, charges.begin() + end);
    const std::vector<double> sequential_dipoles(dipoles.begin() + begin * 3,
                                                 dipoles.begin() + end * 3);
    const std::vector<double> sequential_quadrupoles(quadrupoles.begin() + begin * 6,
                                                     quadrupoles.begin() + end * 6);
    Fixture sequential;
    CHECK(make_fixture(sequential_offsets, sequential_numbers, sequential_positions, sequential,
                       error));
    std::vector<double> sequential_charge_potential(atom_count);
    std::vector<double> sequential_dipole_potential(atom_count * 3u);
    std::vector<double> sequential_quadrupole_potential(atom_count * 6u);
    std::vector<double> sequential_energy(1, 0.0);
    std::vector<double> sequential_gradient(atom_count * 3u, 0.0);
    std::vector<double> sequential_cn(atom_count, 0.0);
    CHECK(xtbloom::detail::gfn2::evaluate_aes2_potential_cpu(
              sequential.plan, sequential.cache, sequential_charges.data(),
              sequential_dipoles.data(), sequential_quadrupoles.data(),
              sequential_charge_potential.data(), sequential_dipole_potential.data(),
              sequential_quadrupole_potential.data(), sequential.workspace,
              error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(energy(sequential, sequential_charges, sequential_dipoles, sequential_quadrupoles,
                 sequential_energy, error));
    CHECK(vjp(sequential, sequential_positions, sequential.coordination_numbers, kGeneration,
              sequential_charges, sequential_dipoles, sequential_quadrupoles, sequential_gradient,
              sequential_cn, error));
    CHECK(batch_energy[system] == sequential_energy[0]);
    for (std::size_t atom = 0; atom < atom_count; ++atom) {
      CHECK(batch_charge_potential[static_cast<std::size_t>(begin) + atom] ==
            sequential_charge_potential[atom]);
    }
    for (std::size_t component = 0; component < atom_count * 3u; ++component) {
      CHECK(batch_dipole_potential[static_cast<std::size_t>(begin) * 3u + component] ==
            sequential_dipole_potential[component]);
    }
    for (std::size_t component = 0; component < atom_count * 6u; ++component) {
      CHECK(batch_quadrupole_potential[static_cast<std::size_t>(begin) * 6u + component] ==
            sequential_quadrupole_potential[component]);
    }
    for (std::size_t coordinate = 0; coordinate < atom_count * 3u; ++coordinate) {
      CHECK(batch_gradient[static_cast<std::size_t>(begin) * 3u + coordinate] ==
            sequential_gradient[coordinate]);
    }
    for (std::size_t atom = 0; atom < atom_count; ++atom) {
      CHECK(batch_cn[static_cast<std::size_t>(begin) + atom] == sequential_cn[atom]);
    }
    const std::int64_t pair_begin = batch.plan.pair_offsets()[system] * 5;
    const std::int64_t pair_end = batch.plan.pair_offsets()[system + 1u] * 5;
    CHECK(std::equal(batch.pair_data.begin() + pair_begin, batch.pair_data.begin() + pair_end,
                     sequential.pair_data.begin()));
  }
  return 0;
}

int test_system_potential_matches_batch_and_failure_isolation() {
  const std::vector<std::int64_t> offsets{0, 2, 2, 5, 6};
  const std::vector<std::int32_t> numbers{3, 1, 8, 6, 1, 79};
  const std::vector<double> positions{
      0.0, 0.0, -1.2, 0.1, 0.2, 1.4, -0.7, 0.5, 0.1, 1.2, -0.8, 0.9, 2.1, 0.4, -1.0, 3.3, -0.6, 0.7,
  };
  Fixture batch;
  std::string error;
  CHECK(make_fixture(offsets, numbers, positions, batch, error));
  std::vector<double> charges(numbers.size());
  std::vector<double> dipoles(numbers.size() * 3u);
  std::vector<double> quadrupoles(numbers.size() * 6u);
  for (std::size_t atom = 0; atom < numbers.size(); ++atom) {
    charges[atom] = 0.08 * static_cast<double>(atom + 1u) - 0.27;
    for (std::size_t component = 0; component < 3u; ++component) {
      dipoles[atom * 3u + component] =
          0.03 * static_cast<double>(atom * 3u + component + 1u) - 0.19;
    }
    const double xx = 0.02 * static_cast<double>(atom + 1u);
    const double yy = -0.015 * static_cast<double>(atom + 2u);
    quadrupoles[atom * 6u] = xx;
    quadrupoles[atom * 6u + 1u] = -0.01 * static_cast<double>(atom + 1u);
    quadrupoles[atom * 6u + 2u] = yy;
    quadrupoles[atom * 6u + 3u] = 0.012 * static_cast<double>(atom + 1u);
    quadrupoles[atom * 6u + 4u] = -0.008 * static_cast<double>(atom + 2u);
    quadrupoles[atom * 6u + 5u] = -xx - yy;
  }
  std::vector<double> batch_charge(numbers.size());
  std::vector<double> batch_dipole(dipoles.size());
  std::vector<double> batch_quadrupole(quadrupoles.size());
  CHECK(xtbloom::detail::gfn2::evaluate_aes2_potential_cpu(
            batch.plan, batch.cache, charges.data(), dipoles.data(), quadrupoles.data(),
            batch_charge.data(), batch_dipole.data(), batch_quadrupole.data(), batch.workspace,
            error) == XTBLOOM_STATUS_SUCCESS);

  /* The one-system primitive reproduces the batch potential slices, including
   * an empty ragged member. */
  std::vector<double> system_charge(charges.size(), -7.0);
  std::vector<double> system_dipole(dipoles.size(), -7.0);
  std::vector<double> system_quadrupole(quadrupoles.size(), -7.0);
  for (std::int64_t system = 0; system < batch.plan.batch_size(); ++system) {
    CHECK(xtbloom::detail::gfn2::evaluate_aes2_potential_system_cpu(
              batch.plan, batch.cache, system, charges.data(), dipoles.data(), quadrupoles.data(),
              system_charge.data(), system_dipole.data(), system_quadrupole.data(), batch.workspace,
              error) == XTBLOOM_STATUS_SUCCESS);
    const std::int64_t atom_begin = offsets[system];
    const std::int64_t atom_end = offsets[system + 1u];
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
      CHECK(system_charge[static_cast<std::size_t>(atom)] ==
            batch_charge[static_cast<std::size_t>(atom)]);
      for (std::size_t component = 0; component < 3u; ++component) {
        CHECK(system_dipole[static_cast<std::size_t>(atom) * 3u + component] ==
              batch_dipole[static_cast<std::size_t>(atom) * 3u + component]);
      }
      for (std::size_t component = 0; component < 6u; ++component) {
        CHECK(system_quadrupole[static_cast<std::size_t>(atom) * 6u + component] ==
              batch_quadrupole[static_cast<std::size_t>(atom) * 6u + component]);
      }
    }
  }

  /* Peer poison must be invisible to the selected system, including a peer
   * cache slice and peer multipoles. */
  const std::size_t peer_atom = 2u;
  const std::size_t peer_pair = static_cast<std::size_t>(batch.plan.pair_offsets()[2]) * 5u;
  const double saved_charge = charges[peer_atom];
  const double saved_pair = batch.pair_data[peer_pair];
  charges[peer_atom] = std::numeric_limits<double>::quiet_NaN();
  batch.pair_data[peer_pair] = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> expected_charge(batch_charge);
  const std::vector<double> expected_dipole(batch_dipole);
  const std::vector<double> expected_quadrupole(batch_quadrupole);
  std::fill(system_charge.begin(), system_charge.end(), 0.0);
  std::fill(system_dipole.begin(), system_dipole.end(), 0.0);
  std::fill(system_quadrupole.begin(), system_quadrupole.end(), 0.0);
  CHECK(xtbloom::detail::gfn2::evaluate_aes2_potential_system_cpu(
            batch.plan, batch.cache, 0, charges.data(), dipoles.data(), quadrupoles.data(),
            system_charge.data(), system_dipole.data(), system_quadrupole.data(), batch.workspace,
            error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t atom = 0; atom < 2u; ++atom) {
    CHECK(system_charge[atom] == expected_charge[atom]);
    for (std::size_t component = 0; component < 3u; ++component) {
      CHECK(system_dipole[atom * 3u + component] == expected_dipole[atom * 3u + component]);
    }
    for (std::size_t component = 0; component < 6u; ++component) {
      CHECK(system_quadrupole[atom * 6u + component] == expected_quadrupole[atom * 6u + component]);
    }
  }
  charges[peer_atom] = saved_charge;
  batch.pair_data[peer_pair] = saved_pair;

  /* Selecting the poisoned member reports an isolated target failure and
   * leaves its output slices unchanged. */
  std::fill(system_charge.begin(), system_charge.end(), 1.5);
  const std::vector<double> untouched_charge(system_charge);
  charges[static_cast<std::size_t>(offsets[2])] = std::numeric_limits<double>::infinity();
  CHECK(xtbloom::detail::gfn2::evaluate_aes2_potential_system_cpu(
            batch.plan, batch.cache, 2, charges.data(), dipoles.data(), quadrupoles.data(),
            system_charge.data(), system_dipole.data(), system_quadrupole.data(), batch.workspace,
            error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(system_charge == untouched_charge);
  charges[static_cast<std::size_t>(offsets[2])] = 0.08 * 3.0 - 0.27;

  /* Out-of-range system and foreign cache are structural failures. */
  CHECK(xtbloom::detail::gfn2::evaluate_aes2_potential_system_cpu(
            batch.plan, batch.cache, -1, charges.data(), dipoles.data(), quadrupoles.data(),
            system_charge.data(), system_dipole.data(), system_quadrupole.data(), batch.workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::evaluate_aes2_potential_system_cpu(
            batch.plan, batch.cache, batch.plan.batch_size(), charges.data(), dipoles.data(),
            quadrupoles.data(), system_charge.data(), system_dipole.data(),
            system_quadrupole.data(), batch.workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  Fixture foreign;
  CHECK(make_fixture(offsets, numbers, positions, foreign, error));
  CHECK(xtbloom::detail::gfn2::evaluate_aes2_potential_system_cpu(
            batch.plan, foreign.cache, 0, charges.data(), dipoles.data(), quadrupoles.data(),
            system_charge.data(), system_dipole.data(), system_quadrupole.data(), batch.workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  /* No per-call allocation in steady state. */
  const std::size_t before = allocation_test::count.load(std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const xtbloom_status_t status = xtbloom::detail::gfn2::evaluate_aes2_potential_system_cpu(
      batch.plan, batch.cache, 0, charges.data(), dipoles.data(), quadrupoles.data(),
      system_charge.data(), system_dipole.data(), system_quadrupole.data(), batch.workspace, error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  CHECK(status == XTBLOOM_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == before);
  return 0;
}

int test_failure_atomicity_and_plan_identity() {
  const std::vector<std::int64_t> offsets{0, 2, 4};
  const std::vector<std::int32_t> numbers{3, 1, 8, 6};
  std::vector<double> positions{0.0, 0.0, -1.3, 0.2, 0.1, 1.4, -0.8, 0.5, 0.2, 1.1, -0.7, 0.9};
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(offsets, numbers, positions, fixture, error));
  const std::vector<double> original_pairs = fixture.pair_data;
  const AES2GeometryCache original_cache = fixture.cache;
  const std::vector<double> original_coordination_numbers = fixture.coordination_numbers;

  fixture.coordination_numbers.back() = std::numeric_limits<double>::quiet_NaN();
  CHECK(xtbloom::detail::gfn2::update_aes2_geometry_cache_cpu(
            fixture.plan, positions.data(), fixture.coordination_numbers.data(), 99u,
            fixture.pair_data.data(), fixture.pair_data.size(), fixture.workspace, fixture.cache,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.pair_data == original_pairs);
  CHECK(same_cache(fixture.cache, original_cache));
  fixture.coordination_numbers.back() = original_coordination_numbers.back();

  positions[9] = positions[6];
  positions[10] = positions[7];
  positions[11] = positions[8];
  CHECK(xtbloom::detail::gfn2::update_aes2_geometry_cache_cpu(
            fixture.plan, positions.data(), fixture.coordination_numbers.data(), 100u,
            fixture.pair_data.data(), fixture.pair_data.size(), fixture.workspace, fixture.cache,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.pair_data == original_pairs);
  CHECK(same_cache(fixture.cache, original_cache));

  const std::vector<double> charges{0.2, -0.1, 0.17, -0.27};
  const std::vector<double> dipoles{0.1,  -0.2, 0.3,   -0.1, 0.05,  0.07,
                                    0.08, 0.02, -0.03, 0.04, -0.06, 0.09};
  std::vector<double> quadrupoles{
      0.1,  0.02,  -0.04, 0.01, -0.03, -0.06, -0.03, 0.01, 0.07, -0.02, 0.04, -0.04,
      0.05, -0.03, -0.02, 0.04, 0.01,  -0.03, -0.08, 0.02, 0.03, -0.01, 0.05, 0.05,
  };
  std::vector<double> charge_potential(4, 7.0);
  std::vector<double> dipole_potential(12, 8.0);
  std::vector<double> quadrupole_potential(24, 9.0);
  const std::vector<double> original_charge_potential = charge_potential;
  const std::vector<double> original_dipole_potential = dipole_potential;
  const std::vector<double> original_quadrupole_potential = quadrupole_potential;
  quadrupoles.back() = std::numeric_limits<double>::infinity();
  CHECK(xtbloom::detail::gfn2::evaluate_aes2_potential_cpu(
            fixture.plan, fixture.cache, charges.data(), dipoles.data(), quadrupoles.data(),
            charge_potential.data(), dipole_potential.data(), quadrupole_potential.data(),
            fixture.workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(charge_potential == original_charge_potential);
  CHECK(dipole_potential == original_dipole_potential);
  CHECK(quadrupole_potential == original_quadrupole_potential);
  quadrupoles.back() = 0.05;

  std::vector<double> energies{1.0, 2.0};
  const std::vector<double> original_energies = energies;
  const double saved_kernel = fixture.pair_data.back();
  fixture.pair_data.back() = std::numeric_limits<double>::quiet_NaN();
  CHECK(xtbloom::detail::gfn2::add_aes2_energy_cpu(
            fixture.plan, fixture.cache, charges.data(), dipoles.data(), quadrupoles.data(),
            energies.data(), fixture.workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energies == original_energies);
  fixture.pair_data.back() = saved_kernel;

  double& aliased_energy = const_cast<double&>(charges[0]);
  const double aliased_energy_before = aliased_energy;
  CHECK(xtbloom::detail::gfn2::add_aes2_energy_system_cpu(
            fixture.plan, fixture.cache, 0, charges.data(), dipoles.data(), quadrupoles.data(),
            aliased_energy, fixture.workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(aliased_energy == aliased_energy_before);
  double system_energy = 3.0;
  AES2Workspace short_energy_workspace = fixture.workspace;
  --short_energy_workspace.batch_elements;
  CHECK(xtbloom::detail::gfn2::add_aes2_energy_system_cpu(
            fixture.plan, fixture.cache, 0, charges.data(), dipoles.data(), quadrupoles.data(),
            system_energy, short_energy_workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(system_energy == 3.0);
  CHECK(xtbloom::detail::gfn2::add_aes2_energy_system_cpu(
            fixture.plan, fixture.cache, fixture.plan.batch_size(), charges.data(), dipoles.data(),
            quadrupoles.data(), system_energy, fixture.workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(system_energy == 3.0);

  Fixture reverse;
  const std::vector<std::int32_t> reverse_numbers{8, 6, 3, 1};
  const std::vector<double> reverse_positions{-0.8, 0.5, 0.2,  1.1, -0.7, 0.9,
                                              0.0,  0.0, -1.3, 0.2, 0.1,  1.4};
  CHECK(make_fixture(offsets, reverse_numbers, reverse_positions, reverse, error));
  AES2GeometryCache wrong_cache = fixture.cache;
  wrong_cache.plan_identity = reverse.plan.identity();
  CHECK(xtbloom::detail::gfn2::evaluate_aes2_potential_cpu(
            fixture.plan, wrong_cache, charges.data(), dipoles.data(), quadrupoles.data(),
            charge_potential.data(), dipole_potential.data(), quadrupole_potential.data(),
            fixture.workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(charge_potential == original_charge_potential);

  AES2Workspace short_workspace = fixture.workspace;
  short_workspace.potential_elements -= 1;
  CHECK(xtbloom::detail::gfn2::evaluate_aes2_potential_cpu(
            fixture.plan, fixture.cache, charges.data(), dipoles.data(), quadrupoles.data(),
            charge_potential.data(), dipole_potential.data(), quadrupole_potential.data(),
            short_workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(charge_potential == original_charge_potential);

  CHECK(xtbloom::detail::gfn2::evaluate_aes2_potential_cpu(
            fixture.plan, fixture.cache, charges.data(), dipoles.data(), quadrupoles.data(),
            const_cast<double*>(charges.data()), dipole_potential.data(),
            quadrupole_potential.data(), fixture.workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  positions = {0.0, 0.0, -1.3, 0.2, 0.1, 1.4, -0.8, 0.5, 0.2, 1.1, -0.7, 0.9};
  fixture.coordination_numbers = original_coordination_numbers;
  std::vector<double> gradients(positions.size(), 11.0);
  std::vector<double> coordination_adjoints(numbers.size(), 12.0);
  const std::vector<double> original_gradients = gradients;
  const std::vector<double> original_coordination_adjoints = coordination_adjoints;
  CHECK(xtbloom::detail::gfn2::add_aes2_vjp_cpu(
            fixture.plan, fixture.cache, positions.data(), fixture.coordination_numbers.data(),
            kGeneration + 1u, charges.data(), dipoles.data(), quadrupoles.data(), gradients.data(),
            coordination_adjoints.data(), fixture.workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(gradients == original_gradients);
  CHECK(coordination_adjoints == original_coordination_adjoints);

  positions[0] += 0.01;
  CHECK(xtbloom::detail::gfn2::add_aes2_vjp_cpu(
            fixture.plan, fixture.cache, positions.data(), fixture.coordination_numbers.data(),
            kGeneration, charges.data(), dipoles.data(), quadrupoles.data(), gradients.data(),
            coordination_adjoints.data(), fixture.workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(gradients == original_gradients);
  CHECK(coordination_adjoints == original_coordination_adjoints);
  positions[0] -= 0.01;

  fixture.coordination_numbers[1] += 10.0;
  CHECK(xtbloom::detail::gfn2::add_aes2_vjp_cpu(
            fixture.plan, fixture.cache, positions.data(), fixture.coordination_numbers.data(),
            kGeneration, charges.data(), dipoles.data(), quadrupoles.data(), gradients.data(),
            coordination_adjoints.data(), fixture.workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(gradients == original_gradients);
  CHECK(coordination_adjoints == original_coordination_adjoints);
  fixture.coordination_numbers[1] = original_coordination_numbers[1];

  quadrupoles.back() = std::numeric_limits<double>::infinity();
  CHECK(xtbloom::detail::gfn2::add_aes2_vjp_cpu(
            fixture.plan, fixture.cache, positions.data(), fixture.coordination_numbers.data(),
            kGeneration, charges.data(), dipoles.data(), quadrupoles.data(), gradients.data(),
            coordination_adjoints.data(), fixture.workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(gradients == original_gradients);
  CHECK(coordination_adjoints == original_coordination_adjoints);
  quadrupoles.back() = 0.05;

  AES2Workspace short_vjp_workspace = fixture.workspace;
  short_vjp_workspace.coordination_elements -= 1;
  CHECK(xtbloom::detail::gfn2::add_aes2_vjp_cpu(
            fixture.plan, fixture.cache, positions.data(), fixture.coordination_numbers.data(),
            kGeneration, charges.data(), dipoles.data(), quadrupoles.data(), gradients.data(),
            coordination_adjoints.data(), short_vjp_workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(gradients == original_gradients);
  CHECK(coordination_adjoints == original_coordination_adjoints);

  CHECK(xtbloom::detail::gfn2::add_aes2_vjp_cpu(
            fixture.plan, fixture.cache, positions.data(), fixture.coordination_numbers.data(),
            kGeneration, charges.data(), dipoles.data(), quadrupoles.data(),
            const_cast<double*>(dipoles.data()), coordination_adjoints.data(), fixture.workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(coordination_adjoints == original_coordination_adjoints);

  AES2Workspace aliased_vjp_workspace = fixture.workspace;
  aliased_vjp_workspace.coordination_scratch = coordination_adjoints.data();
  CHECK(xtbloom::detail::gfn2::add_aes2_vjp_cpu(
            fixture.plan, fixture.cache, positions.data(), fixture.coordination_numbers.data(),
            kGeneration, charges.data(), dipoles.data(), quadrupoles.data(), gradients.data(),
            coordination_adjoints.data(), aliased_vjp_workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(gradients == original_gradients);
  CHECK(coordination_adjoints == original_coordination_adjoints);
  return 0;
}

int test_zero_steady_state_allocations() {
  const std::vector<std::int64_t> offsets{0, 3, 5};
  const std::vector<std::int32_t> numbers{8, 1, 1, 6, 1};
  const std::vector<double> positions{-0.8, 0.3, 0.2, 0.9, -0.6, 0.7,  0.4, 1.3,
                                      -0.9, 2.0, 0.5, 0.1, 3.2,  -0.4, 0.8};
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(offsets, numbers, positions, fixture, error));
  std::vector<double> charges(numbers.size(), 0.1);
  std::vector<double> dipoles(numbers.size() * 3u, -0.03);
  std::vector<double> quadrupoles(numbers.size() * 6u, 0.0);
  for (std::size_t atom = 0; atom < numbers.size(); ++atom) {
    quadrupoles[atom * 6u] = 0.04;
    quadrupoles[atom * 6u + 2u] = -0.01;
    quadrupoles[atom * 6u + 5u] = -0.03;
  }
  std::vector<double> charge_potential(numbers.size());
  std::vector<double> dipole_potential(dipoles.size());
  std::vector<double> quadrupole_potential(quadrupoles.size());
  std::vector<double> energies(2, 0.0);
  std::vector<double> gradients(positions.size(), 0.0);
  std::vector<double> coordination_adjoints(numbers.size(), 0.0);

  CHECK(xtbloom::detail::gfn2::update_aes2_geometry_cache_cpu(
            fixture.plan, positions.data(), fixture.coordination_numbers.data(), kGeneration + 1u,
            fixture.pair_data.data(), fixture.pair_data.size(), fixture.workspace, fixture.cache,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::evaluate_aes2_potential_cpu(
            fixture.plan, fixture.cache, charges.data(), dipoles.data(), quadrupoles.data(),
            charge_potential.data(), dipole_potential.data(), quadrupole_potential.data(),
            fixture.workspace, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(energy(fixture, charges, dipoles, quadrupoles, energies, error));
  CHECK(vjp(fixture, positions, fixture.coordination_numbers, kGeneration + 1u, charges, dipoles,
            quadrupoles, gradients, coordination_adjoints, error));

  allocation_test::count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const xtbloom_status_t update_status = xtbloom::detail::gfn2::update_aes2_geometry_cache_cpu(
      fixture.plan, positions.data(), fixture.coordination_numbers.data(), kGeneration + 2u,
      fixture.pair_data.data(), fixture.pair_data.size(), fixture.workspace, fixture.cache, error);
  const xtbloom_status_t potential_status = xtbloom::detail::gfn2::evaluate_aes2_potential_cpu(
      fixture.plan, fixture.cache, charges.data(), dipoles.data(), quadrupoles.data(),
      charge_potential.data(), dipole_potential.data(), quadrupole_potential.data(),
      fixture.workspace, error);
  const xtbloom_status_t energy_status = xtbloom::detail::gfn2::add_aes2_energy_cpu(
      fixture.plan, fixture.cache, charges.data(), dipoles.data(), quadrupoles.data(),
      energies.data(), fixture.workspace, error);
  double system_energy = 0.0;
  const xtbloom_status_t system_energy_status = xtbloom::detail::gfn2::add_aes2_energy_system_cpu(
      fixture.plan, fixture.cache, 1, charges.data(), dipoles.data(), quadrupoles.data(),
      system_energy, fixture.workspace, error);
  const xtbloom_status_t vjp_status = xtbloom::detail::gfn2::add_aes2_vjp_cpu(
      fixture.plan, fixture.cache, positions.data(), fixture.coordination_numbers.data(),
      kGeneration + 2u, charges.data(), dipoles.data(), quadrupoles.data(), gradients.data(),
      coordination_adjoints.data(), fixture.workspace, error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  CHECK(update_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(potential_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(energy_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(system_energy_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(vjp_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == 0u);
  return 0;
}

}  // namespace

int main() {
  if (const int line = test_dxtb_oracles_and_isolated_components(); line != 0) {
    return line;
  }
  if (const int line = test_coordinate_and_cn_vjp_finite_differences(); line != 0) {
    return line;
  }
  if (const int line = test_energy_potential_consistency(); line != 0) {
    return line;
  }
  if (const int line = test_rotation_covariance_and_tracelessness(); line != 0) {
    return line;
  }
  if (const int line = test_ragged_matches_sequential(); line != 0) {
    return line;
  }
  if (const int line = test_system_potential_matches_batch_and_failure_isolation(); line != 0) {
    return line;
  }
  if (const int line = test_failure_atomicity_and_plan_identity(); line != 0) {
    return line;
  }
  return test_zero_steady_state_allocations();
}
