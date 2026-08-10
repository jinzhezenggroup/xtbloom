#include "model/gfn2/d4.hpp"

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
#include <vector>

namespace allocation_test {
std::atomic<bool> enabled{false};
std::atomic<std::size_t> count{0u};
}  // namespace allocation_test

void* operator new(std::size_t size) {
  if (allocation_test::enabled.load(std::memory_order_relaxed)) {
    allocation_test::count.fetch_add(1u, std::memory_order_relaxed);
  }
  if (void* pointer = std::malloc(size == 0u ? 1u : size)) {
    return pointer;
  }
  throw std::bad_alloc();
}

void* operator new[](std::size_t size) { return ::operator new(size); }

void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

using xtbloom::detail::gfn2::D4GeometryCache;
using xtbloom::detail::gfn2::D4Plan;
using xtbloom::detail::gfn2::D4Workspace;

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

struct AlignedWorkspace {
  explicit AlignedWorkspace(std::size_t size) : storage(size + 63u) {
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(storage.data());
    const std::uintptr_t aligned = (address + 63u) & ~std::uintptr_t{63u};
    data = reinterpret_cast<void*>(aligned);
  }

  std::vector<std::byte> storage;
  void* data = nullptr;
};

struct Fixture {
  std::array<std::int64_t, 2> offsets{0, 4};
  std::array<std::int32_t, 4> atomic_numbers{8, 1, 1, 6};
  std::array<double, 12> positions{
      0.0, 0.0, 0.0, 1.43, 0.0, 1.1, -1.43, 0.0, 1.1, 2.7, 1.2, -0.4,
  };
  std::array<double, 4> charges{-0.6, 0.25, 0.25, 0.1};
  D4Plan plan;
  AlignedWorkspace storage{1u};
  D4Workspace workspace;
  std::vector<double> pair_data;
  std::vector<double> coordination;
  D4GeometryCache cache;
  std::uint64_t generation = 1u;

  bool initialize(std::string& error) {
    if (xtbloom::detail::gfn2::make_d4_plan(1, 4, offsets.data(), atomic_numbers.data(), plan,
                                            error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    storage = AlignedWorkspace(plan.workspace_size_bytes());
    if (xtbloom::detail::gfn2::bind_d4_workspace(plan, storage.data, plan.workspace_size_bytes(),
                                                 workspace, error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    pair_data.resize(static_cast<std::size_t>(plan.total_pairs()) *
                     xtbloom::detail::gfn2::kD4PairDataElements);
    coordination.resize(static_cast<std::size_t>(plan.total_atoms()));
    return update(error);
  }

  bool update(std::string& error) {
    return xtbloom::detail::gfn2::update_d4_geometry_cache_cpu(
               plan, positions.data(), generation++, pair_data.data(), pair_data.size(),
               coordination.data(), coordination.size(), workspace, cache,
               error) == XTBLOOM_STATUS_SUCCESS;
  }

  bool two_body_energy(double& energy, std::string& error) {
    std::array<double, 1> energies{};
    std::array<double, 4> potentials{};
    if (!update(error) || xtbloom::detail::gfn2::evaluate_d4_two_body_cpu(
                              plan, cache, charges.data(), energies.data(), potentials.data(),
                              workspace, error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    energy = energies[0];
    return true;
  }

  bool atm_energy(double& energy, std::string& error) {
    std::array<double, 1> energies{};
    if (!update(error) ||
        xtbloom::detail::gfn2::evaluate_d4_atm_cpu(plan, cache, energies.data(), workspace,
                                                   error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    energy = energies[0];
    return true;
  }
};

int test_d4_coordination_and_ragged_batch() {
  constexpr std::array<std::int64_t, 5> offsets{0, 3, 4, 4, 7};
  constexpr std::array<std::int32_t, 7> atomic_numbers{8, 1, 1, 6, 6, 7, 8};
  constexpr std::array<double, 21> positions{
      0.0, 0.0, 0.0, 1.43, 0.0, 1.1, -1.43, 0.0, 1.1, 2.0, 2.0,
      2.0, 0.0, 0.0, 0.0,  2.2, 0.0, 0.0,   4.4, 0.0, 0.0,
  };
  D4Plan plan;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_d4_plan(4, 7, offsets.data(), atomic_numbers.data(), plan,
                                            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(plan.total_pairs() == 6);
  CHECK(plan.matches_atomic_numbers(atomic_numbers.data()));
  CHECK(plan.overlaps_storage(&plan, sizeof(plan)));
  CHECK(plan.overlaps_storage(plan.identity(), 1u));
  CHECK(plan.overlaps_storage(plan.atom_offsets().data(), sizeof(std::int64_t)));
  CHECK(plan.overlaps_storage(plan.pair_offsets().data(), sizeof(std::int64_t)));
  CHECK(plan.overlaps_storage(nullptr, 1u));
  CHECK(!plan.overlaps_storage(positions.data(), sizeof(positions)));
  CHECK(!plan.overlaps_storage(nullptr, 0u));
  auto changed = atomic_numbers;
  changed[0] = 7;
  CHECK(!plan.matches_atomic_numbers(changed.data()));

  AlignedWorkspace storage(plan.workspace_size_bytes());
  D4Workspace workspace;
  CHECK(xtbloom::detail::gfn2::bind_d4_workspace(plan, storage.data, plan.workspace_size_bytes(),
                                                 workspace, error) == XTBLOOM_STATUS_SUCCESS);
  std::vector<double> pair_data(static_cast<std::size_t>(plan.total_pairs()) *
                                xtbloom::detail::gfn2::kD4PairDataElements);
  std::array<double, 7> coordination{};
  D4GeometryCache cache;
  CHECK(xtbloom::detail::gfn2::update_d4_geometry_cache_cpu(
            plan, positions.data(), 1u, pair_data.data(), pair_data.size(), coordination.data(),
            coordination.size(), workspace, cache, error) == XTBLOOM_STATUS_SUCCESS);

  /* Pinned mctc-lib D4 erf-CN semantics: kcn=7.5 and EN weighting. */
  CHECK(near(coordination[0], 1.6117226819127606, 2.0e-15));
  CHECK(near(coordination[1], 0.8058613409563804, 2.0e-15));
  CHECK(near(coordination[2], 0.8058613409563804, 2.0e-15));
  CHECK(coordination[3] == 0.0);
  CHECK(coordination[4] > 0.0);
  CHECK(coordination[5] > coordination[4]);
  CHECK(coordination[6] > 0.0);
  return 0;
}

int test_two_body_charge_and_coordinate_derivatives() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(error));
  std::array<double, 1> energies{};
  std::array<double, 4> potentials{};
  std::array<double, 12> gradients{};
  CHECK(xtbloom::detail::gfn2::evaluate_d4_two_body_cpu(
            fixture.plan, fixture.cache, fixture.charges.data(), energies.data(), potentials.data(),
            fixture.workspace, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::add_d4_two_body_gradient_cpu(
            fixture.plan, fixture.cache, fixture.charges.data(), gradients.data(),
            fixture.workspace, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(near(energies[0], -0.0005923540861122829, 2.0e-18));

  constexpr double step = 1.0e-5;
  for (std::size_t atom = 0; atom < fixture.charges.size(); ++atom) {
    fixture.charges[atom] += step;
    double right = 0.0;
    CHECK(fixture.two_body_energy(right, error));
    fixture.charges[atom] -= 2.0 * step;
    double left = 0.0;
    CHECK(fixture.two_body_energy(left, error));
    fixture.charges[atom] += step;
    CHECK(near((right - left) / (2.0 * step), potentials[atom], 2.0e-11));
  }
  for (std::size_t coordinate = 0; coordinate < fixture.positions.size(); ++coordinate) {
    fixture.positions[coordinate] += step;
    double right = 0.0;
    CHECK(fixture.two_body_energy(right, error));
    fixture.positions[coordinate] -= 2.0 * step;
    double left = 0.0;
    CHECK(fixture.two_body_energy(left, error));
    fixture.positions[coordinate] += step;
    CHECK(near((right - left) / (2.0 * step), gradients[coordinate], 2.0e-10));
  }
  for (std::size_t axis = 0; axis < 3; ++axis) {
    double sum = 0.0;
    for (std::size_t atom = 0; atom < fixture.charges.size(); ++atom) {
      sum += gradients[atom * 3u + axis];
    }
    CHECK(near(sum, 0.0, 2.0e-18));
  }
  return 0;
}

int test_system_two_body_isolation_and_batch_parity() {
  constexpr std::array<std::int64_t, 5> offsets{0, 3, 4, 4, 7};
  constexpr std::array<std::int32_t, 7> atomic_numbers{8, 1, 1, 6, 6, 7, 8};
  constexpr std::array<double, 21> positions{
      0.0, 0.0, 0.0, 1.43, 0.0, 1.1, -1.43, 0.0, 1.1, 2.0, 2.0,
      2.0, 0.0, 0.0, 0.0,  2.2, 0.0, 0.0,   4.4, 0.0, 0.0,
  };
  std::array<double, 7> charges{-0.55, 0.27, 0.28, 0.0, -0.3, 0.1, 0.2};
  D4Plan plan;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_d4_plan(4, 7, offsets.data(), atomic_numbers.data(), plan,
                                            error) == XTBLOOM_STATUS_SUCCESS);
  AlignedWorkspace storage(plan.workspace_size_bytes());
  D4Workspace workspace;
  CHECK(xtbloom::detail::gfn2::bind_d4_workspace(plan, storage.data, plan.workspace_size_bytes(),
                                                 workspace, error) == XTBLOOM_STATUS_SUCCESS);
  std::vector<double> pair_data(static_cast<std::size_t>(plan.total_pairs()) *
                                xtbloom::detail::gfn2::kD4PairDataElements);
  std::array<double, 7> coordination{};
  D4GeometryCache cache;
  CHECK(xtbloom::detail::gfn2::update_d4_geometry_cache_cpu(
            plan, positions.data(), 1u, pair_data.data(), pair_data.size(), coordination.data(),
            coordination.size(), workspace, cache, error) == XTBLOOM_STATUS_SUCCESS);

  std::array<double, 4> batch_energies{};
  std::array<double, 7> batch_potentials{};
  CHECK(xtbloom::detail::gfn2::evaluate_d4_two_body_cpu(
            plan, cache, charges.data(), batch_energies.data(), batch_potentials.data(), workspace,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(batch_energies[1] == 0.0);
  CHECK(batch_energies[2] == 0.0);

  for (std::int64_t system = 0; system < plan.batch_size(); ++system) {
    double system_energy = 41.0;
    std::array<double, 7> system_potentials{};
    system_potentials.fill(73.0);
    CHECK(xtbloom::detail::gfn2::evaluate_d4_two_body_system_cpu(
              plan, cache, system, charges.data(), system_energy, system_potentials.data(),
              workspace, error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(system_energy == batch_energies[static_cast<std::size_t>(system)]);
    const std::int64_t begin = offsets[static_cast<std::size_t>(system)];
    const std::int64_t end = offsets[static_cast<std::size_t>(system) + 1u];
    for (std::int64_t atom = 0; atom < 7; ++atom) {
      const double expected =
          atom >= begin && atom < end ? batch_potentials[static_cast<std::size_t>(atom)] : 73.0;
      CHECK(system_potentials[static_cast<std::size_t>(atom)] == expected);
    }

    double energy_only = -19.0;
    CHECK(xtbloom::detail::gfn2::evaluate_d4_two_body_system_cpu(
              plan, cache, system, charges.data(), energy_only, nullptr, workspace, error) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(energy_only == batch_energies[static_cast<std::size_t>(system)]);
  }

  /* Poison the final system. Both energy+potential and energy-only evaluation
   * of system zero must ignore those peer slices. */
  const std::size_t peer_atom = 4u;
  const std::size_t peer_pair =
      static_cast<std::size_t>(plan.pair_offsets()[3]) * xtbloom::detail::gfn2::kD4PairDataElements;
  const double saved_charge = charges[peer_atom];
  const double saved_coordination = coordination[peer_atom];
  const double saved_pair = pair_data[peer_pair];
  charges[peer_atom] = std::numeric_limits<double>::quiet_NaN();
  coordination[peer_atom] = std::numeric_limits<double>::quiet_NaN();
  pair_data[peer_pair] = std::numeric_limits<double>::quiet_NaN();
  double isolated_energy = 17.0;
  std::array<double, 7> isolated_potentials{};
  isolated_potentials.fill(29.0);
  CHECK(xtbloom::detail::gfn2::evaluate_d4_two_body_system_cpu(
            plan, cache, 0, charges.data(), isolated_energy, isolated_potentials.data(), workspace,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(isolated_energy == batch_energies[0]);
  for (std::size_t atom = 0; atom < isolated_potentials.size(); ++atom) {
    CHECK(isolated_potentials[atom] == (atom < 3u ? batch_potentials[atom] : 29.0));
  }
  double isolated_energy_only = 31.0;
  CHECK(xtbloom::detail::gfn2::evaluate_d4_two_body_system_cpu(
            plan, cache, 0, charges.data(), isolated_energy_only, nullptr, workspace, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(isolated_energy_only == batch_energies[0]);

  double failed_energy = 37.0;
  std::array<double, 7> failed_potentials{};
  failed_potentials.fill(43.0);
  CHECK(xtbloom::detail::gfn2::evaluate_d4_two_body_system_cpu(
            plan, cache, 3, charges.data(), failed_energy, failed_potentials.data(), workspace,
            error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(failed_energy == 37.0);
  CHECK(std::all_of(failed_potentials.begin(), failed_potentials.end(),
                    [](double value) { return value == 43.0; }));
  charges[peer_atom] = saved_charge;
  coordination[peer_atom] = saved_coordination;
  pair_data[peer_pair] = saved_pair;

  const double aliased_charge = charges[0];
  CHECK(xtbloom::detail::gfn2::evaluate_d4_two_body_system_cpu(
            plan, cache, 0, charges.data(), charges[0], failed_potentials.data(), workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(charges[0] == aliased_charge);
  CHECK(xtbloom::detail::gfn2::evaluate_d4_two_body_system_cpu(
            plan, cache, plan.batch_size(), charges.data(), failed_energy, failed_potentials.data(),
            workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(failed_energy == 37.0);
  return 0;
}

int test_atm_gradient_and_charge_independence() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(error));
  std::array<double, 1> energies{};
  std::array<double, 12> gradients{};
  CHECK(xtbloom::detail::gfn2::evaluate_d4_atm_cpu(fixture.plan, fixture.cache, energies.data(),
                                                   fixture.workspace,
                                                   error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::add_d4_atm_gradient_cpu(fixture.plan, fixture.cache,
                                                       gradients.data(), fixture.workspace,
                                                       error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::isfinite(energies[0]));

  constexpr double step = 1.0e-5;
  for (std::size_t coordinate = 0; coordinate < fixture.positions.size(); ++coordinate) {
    fixture.positions[coordinate] += step;
    double right = 0.0;
    CHECK(fixture.atm_energy(right, error));
    fixture.positions[coordinate] -= 2.0 * step;
    double left = 0.0;
    CHECK(fixture.atm_energy(left, error));
    fixture.positions[coordinate] += step;
    CHECK(near((right - left) / (2.0 * step), gradients[coordinate], 2.0e-12));
  }
  return 0;
}

int test_validation_and_zero_allocation_hot_path() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(error));
  std::array<double, 1> energies{};
  std::array<double, 4> potentials{};
  double system_energy = 0.0;
  std::array<double, 4> system_potentials{};
  std::array<double, 12> gradients{};
  CHECK(xtbloom::detail::gfn2::evaluate_d4_two_body_cpu(
            fixture.plan, fixture.cache, fixture.charges.data(), energies.data(), potentials.data(),
            fixture.workspace, error) == XTBLOOM_STATUS_SUCCESS);

  allocation_test::count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const xtbloom_status_t update_status = xtbloom::detail::gfn2::update_d4_geometry_cache_cpu(
      fixture.plan, fixture.positions.data(), fixture.generation++, fixture.pair_data.data(),
      fixture.pair_data.size(), fixture.coordination.data(), fixture.coordination.size(),
      fixture.workspace, fixture.cache, error);
  const xtbloom_status_t energy_status = xtbloom::detail::gfn2::evaluate_d4_two_body_cpu(
      fixture.plan, fixture.cache, fixture.charges.data(), energies.data(), potentials.data(),
      fixture.workspace, error);
  const xtbloom_status_t system_energy_status =
      xtbloom::detail::gfn2::evaluate_d4_two_body_system_cpu(
          fixture.plan, fixture.cache, 0, fixture.charges.data(), system_energy,
          system_potentials.data(), fixture.workspace, error);
  const xtbloom_status_t system_energy_only_status =
      xtbloom::detail::gfn2::evaluate_d4_two_body_system_cpu(fixture.plan, fixture.cache, 0,
                                                             fixture.charges.data(), system_energy,
                                                             nullptr, fixture.workspace, error);
  const xtbloom_status_t gradient_status = xtbloom::detail::gfn2::add_d4_two_body_gradient_cpu(
      fixture.plan, fixture.cache, fixture.charges.data(), gradients.data(), fixture.workspace,
      error);
  const xtbloom_status_t atm_energy_status = xtbloom::detail::gfn2::evaluate_d4_atm_cpu(
      fixture.plan, fixture.cache, energies.data(), fixture.workspace, error);
  const xtbloom_status_t atm_gradient_status = xtbloom::detail::gfn2::add_d4_atm_gradient_cpu(
      fixture.plan, fixture.cache, gradients.data(), fixture.workspace, error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  CHECK(update_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(energy_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(system_energy_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(system_energy_only_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(gradient_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(atm_energy_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(atm_gradient_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == 0u);

  auto invalid_positions = fixture.positions;
  invalid_positions[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK(xtbloom::detail::gfn2::update_d4_geometry_cache_cpu(
            fixture.plan, invalid_positions.data(), fixture.generation++, fixture.pair_data.data(),
            fixture.pair_data.size(), fixture.coordination.data(), fixture.coordination.size(),
            fixture.workspace, fixture.cache, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  const auto positions_before = fixture.positions;
  const auto pair_before = fixture.pair_data;
  const auto coordination_before = fixture.coordination;
  const D4GeometryCache cache_before = fixture.cache;
  CHECK(xtbloom::detail::gfn2::update_d4_geometry_cache_cpu(
            fixture.plan, fixture.positions.data(), fixture.generation++, fixture.pair_data.data(),
            fixture.pair_data.size(), fixture.positions.data(), fixture.positions.size(),
            fixture.workspace, fixture.cache, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.positions == positions_before);
  CHECK(fixture.pair_data == pair_before);
  CHECK(fixture.coordination == coordination_before);
  CHECK(fixture.cache.pair_data == cache_before.pair_data);
  CHECK(fixture.cache.coordination_numbers == cache_before.coordination_numbers);
  CHECK(fixture.cache.geometry_generation == cache_before.geometry_generation);

  D4Workspace malformed_workspace = fixture.workspace;
  ++malformed_workspace.weights;
  energies.fill(17.0);
  potentials.fill(19.0);
  CHECK(xtbloom::detail::gfn2::evaluate_d4_two_body_cpu(
            fixture.plan, fixture.cache, fixture.charges.data(), energies.data(), potentials.data(),
            malformed_workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energies[0] == 17.0);
  CHECK(std::all_of(potentials.begin(), potentials.end(),
                    [](double value) { return value == 19.0; }));

  std::array<double, 4> overlapping_outputs{23.0, 23.0, 23.0, 23.0};
  CHECK(xtbloom::detail::gfn2::evaluate_d4_two_body_cpu(
            fixture.plan, fixture.cache, fixture.charges.data(), overlapping_outputs.data(),
            overlapping_outputs.data(), fixture.workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(overlapping_outputs.begin(), overlapping_outputs.end(),
                    [](double value) { return value == 23.0; }));

  const auto atom_offsets_before = fixture.plan.atom_offsets();
  auto* plan_storage =
      reinterpret_cast<double*>(const_cast<std::int64_t*>(fixture.plan.atom_offsets().data()));
  CHECK(xtbloom::detail::gfn2::evaluate_d4_two_body_cpu(
            fixture.plan, fixture.cache, fixture.charges.data(), plan_storage, potentials.data(),
            fixture.workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.plan.atom_offsets() == atom_offsets_before);

  CHECK(xtbloom::detail::gfn2::evaluate_d4_two_body_cpu(
            fixture.plan, fixture.cache, fixture.workspace.weights, energies.data(),
            potentials.data(), fixture.workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  std::array<double, 12> charge_gradient_alias{};
  std::copy(fixture.charges.begin(), fixture.charges.end(), charge_gradient_alias.begin());
  CHECK(xtbloom::detail::gfn2::add_d4_two_body_gradient_cpu(
            fixture.plan, fixture.cache, charge_gradient_alias.data(), charge_gradient_alias.data(),
            fixture.workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  CHECK(xtbloom::detail::gfn2::evaluate_d4_atm_cpu(fixture.plan, fixture.cache,
                                                   fixture.coordination.data(), fixture.workspace,
                                                   error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::add_d4_atm_gradient_cpu(fixture.plan, fixture.cache,
                                                       fixture.pair_data.data(), fixture.workspace,
                                                       error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  return 0;
}

}  // namespace

int main() {
  if (const int line = test_d4_coordination_and_ragged_batch(); line != 0) {
    return line;
  }
  if (const int line = test_two_body_charge_and_coordinate_derivatives(); line != 0) {
    return line;
  }
  if (const int line = test_system_two_body_isolation_and_batch_parity(); line != 0) {
    return line;
  }
  if (const int line = test_atm_gradient_and_charge_independence(); line != 0) {
    return line;
  }
  return test_validation_and_zero_allocation_hot_path();
}
