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

#include "model/gfn1/halogen.hpp"

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

using xtbloom::detail::gfn1::HalogenPlan;
using xtbloom::detail::gfn1::HalogenWorkspace;

struct AlignedWorkspace {
  void reset(std::size_t size) {
    storage.resize(size + xtbloom::detail::gfn1::kHalogenWorkspaceAlignment - 1u);
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(storage.data());
    const std::uintptr_t mask = xtbloom::detail::gfn1::kHalogenWorkspaceAlignment - 1u;
    data = reinterpret_cast<void*>((address + mask) & ~mask);
  }

  std::vector<std::byte> storage;
  void* data = nullptr;
};

struct Evaluation {
  HalogenPlan plan;
  AlignedWorkspace storage;
  HalogenWorkspace workspace;

  bool initialize(std::int64_t batch_size, std::int64_t total_atoms, const std::int64_t* offsets,
                  const std::int32_t* atomic_numbers, std::string& error) {
    if (xtbloom::detail::gfn1::make_halogen_plan(batch_size, total_atoms, offsets, atomic_numbers,
                                                 plan, error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    storage.reset(plan.workspace_size_bytes());
    return xtbloom::detail::gfn1::bind_halogen_workspace(plan, storage.data,
                                                         plan.workspace_size_bytes(), workspace,
                                                         error) == XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t add(const double* positions, double* energies, double* forces,
                       std::string& error) const {
    return xtbloom::detail::gfn1::add_halogen_cpu(plan, positions, energies, forces, workspace,
                                                  error);
  }
};

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

// XTBLOOM_GFN1_FIXTURE_BEGIN gfn1-halogen-tblite-dxtb
int test_tblite_reference_energies_and_gradient() {
  /*
   * Exact nonperiodic fixtures from tblite test/unit/test_halogen.f90 at
   * revision 133f91ef. Coordinates are bohr; the three expected values are
   * the isolated GFN1 classical halogen component in Hartree.
   */
  constexpr std::array<std::int64_t, 2> offsets6{0, 6};
  constexpr std::array<std::int32_t, 6> br2nh3_numbers{35, 35, 7, 1, 1, 1};
  constexpr std::array<double, 18> br2nh3_positions{
      0.0,          0.0,          3.114952513,  0.0,         0.0,          -1.256718806,
      0.0,          0.0,          -6.302011301, 0.0,         1.787127097,  -6.9747084,
      -1.547696925, -0.893562604, -6.9747084,   1.547696925, -0.893562604, -6.9747084,
  };

  Evaluation evaluation;
  std::string error;
  CHECK(evaluation.initialize(1, 6, offsets6.data(), br2nh3_numbers.data(), error));
  std::array<double, 1> energy{};
  std::array<double, 18> forces{};
  CHECK(evaluation.add(br2nh3_positions.data(), energy.data(), forces.data(), error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(near(energy[0], 2.4763110097465683e-3, 3.0e-17));
  /* dxtb's pinned sample publishes gradients; xTBloom publishes forces. */
  CHECK(near(forces[5], 1.3105758671001298e-2, 3.0e-15));
  CHECK(near(forces[8], -1.3105758671001298e-2, 3.0e-15));
  for (std::size_t coordinate = 0; coordinate < forces.size(); ++coordinate) {
    if (coordinate != 5u && coordinate != 8u) {
      CHECK(near(forces[coordinate], 0.0, 2.0e-16));
    }
  }

  constexpr std::array<std::int32_t, 6> br2och2_numbers{35, 35, 8, 6, 1, 1};
  std::vector<double> br2och2_positions{
      -1.785333747, -3.126082999, 0.0, 0.0,         0.816042264, 0.0, 2.658286999, 5.297075806, 0.0,
      4.885971586,  4.861161373,  0.0, 5.615509753, 2.908222159, 0.0, 6.289076126, 6.399636435, 0.0,
  };
  CHECK(evaluation.initialize(1, 6, offsets6.data(), br2och2_numbers.data(), error));
  energy = {0.0};
  forces.fill(0.0);
  CHECK(evaluation.add(br2och2_positions.data(), energy.data(), forces.data(), error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(near(energy[0], -6.7587305781592112e-4, 3.0e-17));

  constexpr std::array<double, 3> steps{1.0e-4, 2.0e-5, 4.0e-6};
  constexpr std::array<double, 3> tolerances{2.0e-9, 3.0e-10, 2.0e-9};
  for (std::size_t step_index = 0; step_index < steps.size(); ++step_index) {
    const double step = steps[step_index];
    for (std::size_t coordinate = 0; coordinate < br2och2_positions.size(); ++coordinate) {
      br2och2_positions[coordinate] += step;
      std::array<double, 1> right{};
      CHECK(evaluation.add(br2och2_positions.data(), right.data(), nullptr, error) ==
            XTBLOOM_STATUS_SUCCESS);
      br2och2_positions[coordinate] -= 2.0 * step;
      std::array<double, 1> left{};
      CHECK(evaluation.add(br2och2_positions.data(), left.data(), nullptr, error) ==
            XTBLOOM_STATUS_SUCCESS);
      br2och2_positions[coordinate] += step;
      const double numerical_force = -(right[0] - left[0]) / (2.0 * step);
      CHECK(near(forces[coordinate], numerical_force, tolerances[step_index]));
    }
  }

  std::array<double, 3> net_force{};
  std::array<double, 3> torque{};
  for (std::size_t atom = 0; atom < 6u; ++atom) {
    const double fx = forces[atom * 3u];
    const double fy = forces[atom * 3u + 1u];
    const double fz = forces[atom * 3u + 2u];
    net_force[0] += fx;
    net_force[1] += fy;
    net_force[2] += fz;
    const double x = br2och2_positions[atom * 3u];
    const double y = br2och2_positions[atom * 3u + 1u];
    const double z = br2och2_positions[atom * 3u + 2u];
    torque[0] += y * fz - z * fy;
    torque[1] += z * fx - x * fz;
    torque[2] += x * fy - y * fx;
  }
  for (std::size_t axis = 0; axis < 3u; ++axis) {
    CHECK(near(net_force[axis], 0.0, 3.0e-16));
    CHECK(near(torque[axis], 0.0, 2.0e-15));
  }

  std::array<double, 18> transformed{};
  for (std::size_t atom = 0; atom < 6u; ++atom) {
    const double x = br2och2_positions[atom * 3u];
    const double y = br2och2_positions[atom * 3u + 1u];
    const double z = br2och2_positions[atom * 3u + 2u];
    transformed[atom * 3u] = -y + 2.5;
    transformed[atom * 3u + 1u] = x - 3.0;
    transformed[atom * 3u + 2u] = z + 0.75;
  }
  std::array<double, 1> transformed_energy{};
  std::array<double, 18> transformed_forces{};
  CHECK(evaluation.add(transformed.data(), transformed_energy.data(), transformed_forces.data(),
                       error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(near(transformed_energy[0], energy[0], 3.0e-17));
  for (std::size_t atom = 0; atom < 6u; ++atom) {
    CHECK(near(transformed_forces[atom * 3u], -forces[atom * 3u + 1u], 3.0e-15));
    CHECK(near(transformed_forces[atom * 3u + 1u], forces[atom * 3u], 3.0e-15));
    CHECK(near(transformed_forces[atom * 3u + 2u], forces[atom * 3u + 2u], 3.0e-15));
  }

  constexpr std::array<std::int64_t, 2> offsets5{0, 5};
  constexpr std::array<std::int32_t, 5> finch_numbers{9, 53, 7, 6, 1};
  constexpr std::array<double, 15> finch_positions{
      0.0,          0.0, 4.376378627, 0.0,          0.0, 0.699818447, 0.0,          0.0,
      -4.241811239, 0.0, 0.0,         -6.395206917, 0.0, 0.0,         -8.413872692,
  };
  CHECK(evaluation.initialize(1, 5, offsets5.data(), finch_numbers.data(), error));
  energy = {0.0};
  CHECK(evaluation.add(finch_positions.data(), energy.data(), nullptr, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(near(energy[0], 1.1857937381795408e-2, 8.0e-17));
  return 0;
}
// XTBLOOM_GFN1_FIXTURE_END gfn1-halogen-tblite-dxtb

int test_ragged_topology_cutoff_and_ties() {
  /* Empty and no-halogen peers remain untouched beside one active system. */
  constexpr std::array<std::int64_t, 5> offsets{0, 0, 2, 6, 9};
  constexpr std::array<std::int32_t, 9> numbers{6, 8, 1, 35, 1, 7, 1, 35, 7};
  std::array<double, 27> positions{
      0.0, 0.0, 0.0, 2.0, 0.0,  0.0, -2.0, 0.0, 0.0, 0.0, 0.0,  0.0, 2.0, 0.0,
      0.0, 5.0, 0.0, 0.0, -2.0, 0.0, 0.0,  0.0, 0.0, 0.0, 20.0, 0.0, 0.0,
  };
  Evaluation evaluation;
  std::string error;
  CHECK(evaluation.initialize(4, 9, offsets.data(), numbers.data(), error));
  std::array<double, 4> energies{0.5, 0.5, 0.5, 0.5};
  std::array<double, 27> forces{};
  CHECK(evaluation.add(positions.data(), energies.data(), forces.data(), error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(energies[0] == 0.5 && energies[1] == 0.5);
  CHECK(energies[2] > 0.5);   // lower-index equidistant neighbor lies opposite the acceptor
  CHECK(energies[3] != 0.5);  // equality at the 20-bohr cutoff is included

  positions[24] = std::nextafter(20.0, std::numeric_limits<double>::infinity());
  energies.fill(0.5);
  CHECK(evaluation.add(positions.data(), energies.data(), nullptr, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(energies[3] == 0.5);

  /* Move the lower-index tied neighbor onto the acceptor ray: K-X-A gives h=0. */
  positions[6] = 2.0;
  positions[12] = -2.0;
  energies.fill(0.0);
  CHECK(evaluation.add(positions.data(), energies.data(), nullptr, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(near(energies[2], 0.0, 2.0e-32));

  /* With K=A, the same-ray angular factor and its force are exactly zero. */
  constexpr std::array<std::int64_t, 2> pair_offsets{0, 2};
  constexpr std::array<std::int32_t, 2> pair_numbers{35, 7};
  constexpr std::array<double, 6> pair_positions{0.0, 0.0, 0.0, 3.0, 0.0, 0.0};
  CHECK(evaluation.initialize(1, 2, pair_offsets.data(), pair_numbers.data(), error));
  std::array<double, 1> pair_energy{};
  std::array<double, 6> pair_forces{};
  CHECK(evaluation.add(pair_positions.data(), pair_energy.data(), pair_forces.data(), error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(pair_energy[0] == 0.0);
  CHECK(std::all_of(pair_forces.begin(), pair_forces.end(),
                    [](double value) { return value == 0.0; }));
  return 0;
}

int test_synthetic_radial_value_and_force() {
  constexpr std::array<std::int64_t, 2> offsets{0, 3};
  constexpr std::array<std::int32_t, 3> numbers{1, 35, 7};
  /*
   * The first atom fixes the donor axis opposite the acceptor, so h=1. At
   * r=R0 the radial factor is (1-0.44)/2=0.28 and its derivative is -3/R0.
   * The exact R0 below is the sum of the generated 1.3-scaled Br and N radii.
   */
  constexpr double r0 = 1.3 * (2.210979565805915 + 1.3417055484805127);
  constexpr double strength = 0.0381742;
  constexpr std::array<double, 9> positions{-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, r0, 0.0, 0.0};
  Evaluation evaluation;
  std::string error;
  CHECK(evaluation.initialize(1, 3, offsets.data(), numbers.data(), error));
  std::array<double, 1> energy{};
  std::array<double, 9> forces{};
  CHECK(evaluation.add(positions.data(), energy.data(), forces.data(), error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(near(energy[0], 0.28 * strength, 3.0e-17));
  CHECK(near(forces[6], 3.0 * strength / r0, 3.0e-17));
  CHECK(near(forces[3], -3.0 * strength / r0, 3.0e-17));
  CHECK(forces[0] == 0.0);
  for (std::size_t coordinate = 0; coordinate < forces.size(); ++coordinate) {
    if (coordinate != 3u && coordinate != 6u) {
      CHECK(near(forces[coordinate], 0.0, 2.0e-17));
    }
  }
  return 0;
}

int test_element_sets_and_transactional_validation() {
  constexpr std::array<std::int64_t, 2> offsets{0, 3};
  constexpr std::array<double, 9> positions{-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0};
  Evaluation evaluation;
  std::string error;

  /* F is excluded as a donor and Cl's canonical strength is exactly zero. */
  for (const std::array<std::int32_t, 3> numbers :
       {std::array<std::int32_t, 3>{1, 9, 7}, std::array<std::int32_t, 3>{1, 17, 7}}) {
    CHECK(evaluation.initialize(1, 3, offsets.data(), numbers.data(), error));
    std::array<double, 1> energy{2.0};
    std::array<double, 9> forces{};
    CHECK(evaluation.add(positions.data(), energy.data(), forces.data(), error) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(energy[0] == 2.0);
    CHECK(std::all_of(forces.begin(), forces.end(), [](double value) { return value == 0.0; }));
  }

  /* Every nonzero donor and every reviewed acceptor element is active. */
  for (const std::int32_t donor : {35, 53, 85}) {
    for (const std::int32_t acceptor : {7, 8, 15, 16}) {
      const std::array<std::int32_t, 3> active_numbers{1, donor, acceptor};
      CHECK(evaluation.initialize(1, 3, offsets.data(), active_numbers.data(), error));
      std::array<double, 1> active_energy{};
      CHECK(evaluation.add(positions.data(), active_energy.data(), nullptr, error) ==
            XTBLOOM_STATUS_SUCCESS);
      CHECK(std::isfinite(active_energy[0]) && active_energy[0] != 0.0);
    }
  }

  constexpr std::array<std::int32_t, 3> numbers{1, 35, 7};
  CHECK(evaluation.initialize(1, 3, offsets.data(), numbers.data(), error));
  std::array<double, 1> energy{7.0};
  std::array<double, 9> forces{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  const auto original_energy = energy;
  const auto original_forces = forces;

  std::array<double, 9> bad_positions = positions;
  bad_positions[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK(evaluation.add(bad_positions.data(), energy.data(), forces.data(), error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energy == original_energy && forces == original_forces);

  bad_positions = positions;
  bad_positions[6] = bad_positions[3];
  bad_positions[7] = bad_positions[4];
  bad_positions[8] = bad_positions[5];
  CHECK(evaluation.add(bad_positions.data(), energy.data(), forces.data(), error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energy == original_energy && forces == original_forces);

  CHECK(evaluation.add(positions.data(), const_cast<double*>(positions.data()), nullptr, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(evaluation.add(positions.data(), energy.data(), const_cast<double*>(positions.data()),
                       error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(evaluation.add(positions.data(), energy.data(), energy.data(), error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(energy == original_energy && forces == original_forces);

  alignas(double) std::array<std::byte, sizeof(double) * 2u + 1u> misaligned_storage{};
  double* misaligned = reinterpret_cast<double*>(misaligned_storage.data() + 1u);
  CHECK(evaluation.add(positions.data(), misaligned, nullptr, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);

  energy[0] = std::numeric_limits<double>::infinity();
  CHECK(evaluation.add(positions.data(), energy.data(), forces.data(), error) ==
        XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(std::isinf(energy[0]) && forces == original_forces);
  energy = original_energy;
  forces[4] = std::numeric_limits<double>::quiet_NaN();
  CHECK(evaluation.add(positions.data(), energy.data(), forces.data(), error) ==
        XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(energy == original_energy && std::isnan(forces[4]));

  energy = {std::numeric_limits<double>::max()};
  forces.fill(std::numeric_limits<double>::max());
  CHECK(evaluation.add(positions.data(), energy.data(), forces.data(), error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(energy[0] == std::numeric_limits<double>::max());
  CHECK(std::all_of(forces.begin(), forces.end(),
                    [](double value) { return value == std::numeric_limits<double>::max(); }));

  /* Force accumulation overflows only in caller-owned scratch, after the
   * complete request has been validated. The transactional contract requires
   * the original accumulators to remain byte-for-byte/value unchanged. */
  const std::array<double, 9> overflow_positions{0.0, 1.0e-308, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0};
  energy = original_energy;
  forces = original_forces;
  forces[0] = std::numeric_limits<double>::max();
  const auto overflow_energy_before = energy;
  const auto overflow_forces_before = forces;
  CHECK(evaluation.add(overflow_positions.data(), energy.data(), forces.data(), error) ==
        XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(energy == overflow_energy_before && forces == overflow_forces_before);

  constexpr std::array<std::int32_t, 3> invalid_low{1, 0, 7};
  constexpr std::array<std::int32_t, 3> invalid_high{1, 87, 7};
  constexpr std::array<std::int64_t, 2> bad_start{1, 3};
  constexpr std::array<std::int64_t, 3> descending{0, 3, 2};
  HalogenPlan invalid_plan;
  CHECK(xtbloom::detail::gfn1::make_halogen_plan(1, 3, nullptr, numbers.data(), invalid_plan,
                                                 error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn1::make_halogen_plan(1, 3, bad_start.data(), numbers.data(),
                                                 invalid_plan,
                                                 error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn1::make_halogen_plan(2, 2, descending.data(), numbers.data(),
                                                 invalid_plan,
                                                 error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn1::make_halogen_plan(1, 3, offsets.data(), invalid_low.data(),
                                                 invalid_plan,
                                                 error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn1::make_halogen_plan(1, 3, offsets.data(), invalid_high.data(),
                                                 invalid_plan,
                                                 error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  return 0;
}

int test_sealed_plan_workspace_contract_and_zero_allocation() {
  constexpr std::array<std::int64_t, 2> offsets{0, 4};
  constexpr std::array<std::int32_t, 4> numbers{1, 35, 7, 8};
  constexpr std::array<std::int32_t, 4> changed_numbers{1, 35, 7, 16};
  constexpr std::array<double, 12> positions{
      -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.7, 0.0, 6.0, -0.4, 0.3,
  };
  std::string error;
  Evaluation evaluation;
  CHECK(evaluation.initialize(1, 4, offsets.data(), numbers.data(), error));
  CHECK(evaluation.plan.sealed());
  CHECK(evaluation.plan.batch_size() == 1);
  CHECK(evaluation.plan.total_atoms() == 4);
  CHECK(evaluation.plan.atom_offsets() ==
        std::vector<std::int64_t>(offsets.begin(), offsets.end()));
  CHECK(evaluation.plan.matches_atomic_numbers(numbers.data()));
  CHECK(!evaluation.plan.matches_atomic_numbers(changed_numbers.data()));
  CHECK(evaluation.plan.resident_bytes() > sizeof(HalogenPlan));
  CHECK(evaluation.plan.workspace_size_bytes() > 0u);
  CHECK(evaluation.plan.workspace_size_bytes() %
            xtbloom::detail::gfn1::kHalogenWorkspaceAlignment ==
        0u);
  const auto* plan_offsets = evaluation.plan.atom_offsets().data();
  CHECK(evaluation.plan.overlaps_storage(plan_offsets, sizeof(std::int64_t) * offsets.size()));
  CHECK(!evaluation.plan.overlaps_storage(positions.data(), sizeof(positions)));

  HalogenPlan copied_plan = evaluation.plan;
  CHECK(copied_plan.identity() == evaluation.plan.identity());
  CHECK(copied_plan.matches_atomic_numbers(numbers.data()));
  AlignedWorkspace copied_storage;
  copied_storage.reset(copied_plan.workspace_size_bytes());
  HalogenWorkspace copied_workspace;
  CHECK(xtbloom::detail::gfn1::bind_halogen_workspace(
            copied_plan, copied_storage.data, copied_plan.workspace_size_bytes(), copied_workspace,
            error) == XTBLOOM_STATUS_SUCCESS);

  std::array<double, 1> energy{0.25};
  std::array<double, 12> forces{};
  CHECK(xtbloom::detail::gfn1::add_halogen_cpu(copied_plan, positions.data(), energy.data(),
                                               forces.data(), copied_workspace,
                                               error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(energy[0] != 0.25);

  alignas(xtbloom::detail::gfn1::kHalogenWorkspaceAlignment) std::array<std::byte, 256>
      raw_workspace{};
  HalogenWorkspace unchanged_workspace;
  const HalogenWorkspace workspace_before = unchanged_workspace;
  CHECK(xtbloom::detail::gfn1::bind_halogen_workspace(
            evaluation.plan, raw_workspace.data() + 1u, evaluation.plan.workspace_size_bytes(),
            unchanged_workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(unchanged_workspace.workspace_base == workspace_before.workspace_base);
  CHECK(unchanged_workspace.plan_identity == workspace_before.plan_identity);
  CHECK(xtbloom::detail::gfn1::bind_halogen_workspace(
            evaluation.plan, evaluation.storage.data, evaluation.plan.workspace_size_bytes() - 1u,
            unchanged_workspace, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn1::bind_halogen_workspace(
            evaluation.plan, const_cast<std::int64_t*>(plan_offsets),
            evaluation.plan.workspace_size_bytes(), unchanged_workspace,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  std::array<double, 1> transactional_energy{3.0};
  std::array<double, 12> transactional_forces{};
  transactional_forces.fill(4.0);
  const auto energy_before = transactional_energy;
  const auto forces_before = transactional_forces;
  HalogenWorkspace tampered = evaluation.workspace;
  ++tampered.axis_neighbors;
  CHECK(xtbloom::detail::gfn1::add_halogen_cpu(
            evaluation.plan, positions.data(), transactional_energy.data(),
            transactional_forces.data(), tampered, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(transactional_energy == energy_before && transactional_forces == forces_before);

  auto* plan_energy = reinterpret_cast<double*>(const_cast<std::int64_t*>(plan_offsets));
  CHECK(evaluation.add(positions.data(), plan_energy, nullptr, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  tampered = evaluation.workspace;
  tampered.plan_identity = nullptr;
  CHECK(xtbloom::detail::gfn1::add_halogen_cpu(
            evaluation.plan, positions.data(), transactional_energy.data(),
            transactional_forces.data(), tampered, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(transactional_energy == energy_before && transactional_forces == forces_before);

  Evaluation other;
  CHECK(other.initialize(1, 4, offsets.data(), changed_numbers.data(), error));
  CHECK(xtbloom::detail::gfn1::add_halogen_cpu(evaluation.plan, positions.data(),
                                               transactional_energy.data(),
                                               transactional_forces.data(), other.workspace,
                                               error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(transactional_energy == energy_before && transactional_forces == forces_before);

  auto* workspace_energy = evaluation.workspace.batch_scratch;
  workspace_energy[0] = 6.0;
  CHECK(evaluation.add(positions.data(), workspace_energy, nullptr, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  auto* workspace_forces = evaluation.workspace.force_scratch;
  workspace_forces[0] = 7.0;
  CHECK(evaluation.add(positions.data(), transactional_energy.data(), workspace_forces, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);

  std::array<double, 1> hot_energy{};
  std::array<double, 12> hot_forces{};
  error.reserve(128u);
  allocation_test::count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const xtbloom_status_t hot_status =
      evaluation.add(positions.data(), hot_energy.data(), hot_forces.data(), error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  CHECK(hot_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == 0u);
  return 0;
}

}  // namespace

int main() {
  if (const int line = test_tblite_reference_energies_and_gradient(); line != 0) {
    return line;
  }
  if (const int line = test_ragged_topology_cutoff_and_ties(); line != 0) {
    return line;
  }
  if (const int line = test_synthetic_radial_value_and_force(); line != 0) {
    return line;
  }
  if (const int line = test_element_sets_and_transactional_validation(); line != 0) {
    return line;
  }
  return test_sealed_plan_workspace_contract_and_zero_allocation();
}
