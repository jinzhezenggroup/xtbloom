#include "model/gfn2/es2.hpp"

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

using gpuxtb::detail::gfn2::BasisPlan;
using gpuxtb::detail::gfn2::ES2GeometryCache;
using gpuxtb::detail::gfn2::ES2Plan;
using gpuxtb::detail::gfn2::ES2Workspace;

static_assert(std::is_trivially_copyable_v<ES2GeometryCache>);
static_assert(std::is_standard_layout_v<ES2GeometryCache>);
static_assert(std::is_trivially_copyable_v<ES2Workspace>);
static_assert(std::is_standard_layout_v<ES2Workspace>);
static_assert(std::is_nothrow_copy_constructible_v<ES2Plan>);
static_assert(std::is_nothrow_copy_assignable_v<ES2Plan>);
static_assert(sizeof(ES2Plan) <= 4u * sizeof(void*),
              "ES2Plan must remain a compact constant-time shared handle");

constexpr std::uint64_t kGeometryGeneration = 17u;

struct Evaluation {
  BasisPlan basis;
  ES2Plan plan;
  std::vector<double> matrix;
  std::vector<double> matrix_scratch;
  std::vector<double> shell_scratch;
  std::vector<double> batch_scratch;
  std::vector<double> gradient_scratch;
  ES2Workspace workspace;
  ES2GeometryCache cache;
};

struct MisalignedDoubles {
  static constexpr std::byte kSentinel{0x5a};

  std::vector<std::byte> storage;
  double* data = nullptr;

  explicit MisalignedDoubles(std::size_t count)
      : storage(count * sizeof(double) + 1u, kSentinel),
        data(reinterpret_cast<double*>(storage.data() + 1u)) {}

  bool is_misaligned() const {
    return reinterpret_cast<std::uintptr_t>(data) % alignof(double) != 0u;
  }

  bool untouched() const {
    return std::all_of(storage.begin(), storage.end(),
                       [](std::byte value) { return value == kSentinel; });
  }
};

template <typename Descriptor>
struct alignas(double) DescriptorEnvelope {
  Descriptor descriptor{};
  std::array<std::byte, 64> tail{};
};

bool same_cache_descriptor(const ES2GeometryCache& actual, const ES2GeometryCache& expected) {
  return actual.coulomb_matrix == expected.coulomb_matrix &&
         actual.matrix_elements == expected.matrix_elements &&
         actual.geometry_generation == expected.geometry_generation &&
         actual.plan_identity == expected.plan_identity;
}

bool same_workspace_descriptor(const ES2Workspace& actual, const ES2Workspace& expected) {
  return actual.matrix_scratch == expected.matrix_scratch &&
         actual.matrix_elements == expected.matrix_elements &&
         actual.shell_scratch == expected.shell_scratch &&
         actual.shell_elements == expected.shell_elements &&
         actual.batch_scratch == expected.batch_scratch &&
         actual.batch_elements == expected.batch_elements &&
         actual.gradient_scratch == expected.gradient_scratch &&
         actual.gradient_elements == expected.gradient_elements;
}

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

bool relative_near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance * std::max(std::abs(expected), 1.0e-300);
}

bool make_evaluation(const std::vector<std::int64_t>& atom_offsets,
                     const std::vector<std::int32_t>& atomic_numbers,
                     const std::vector<double>& positions, Evaluation& evaluation,
                     std::string& error) {
  const std::int64_t batch_size = static_cast<std::int64_t>(atom_offsets.size() - 1u);
  if (gpuxtb::detail::gfn2::make_basis_plan(
          batch_size, static_cast<std::int64_t>(atomic_numbers.size()), atom_offsets.data(),
          atomic_numbers.data(), evaluation.basis, error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  if (gpuxtb::detail::gfn2::make_es2_plan(evaluation.basis, atomic_numbers.data(), evaluation.plan,
                                          error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  evaluation.matrix.resize(static_cast<std::size_t>(evaluation.plan.total_matrix_elements()));
  evaluation.matrix_scratch.resize(
      static_cast<std::size_t>(evaluation.plan.total_matrix_elements()));
  evaluation.shell_scratch.resize(static_cast<std::size_t>(evaluation.plan.total_shells()));
  evaluation.batch_scratch.resize(static_cast<std::size_t>(evaluation.plan.batch_size()));
  evaluation.gradient_scratch.resize(static_cast<std::size_t>(evaluation.plan.total_atoms()) * 3u);
  evaluation.workspace = ES2Workspace{
      evaluation.matrix_scratch.data(),   evaluation.plan.total_matrix_elements(),
      evaluation.shell_scratch.data(),    evaluation.plan.total_shells(),
      evaluation.batch_scratch.data(),    evaluation.plan.batch_size(),
      evaluation.gradient_scratch.data(), evaluation.plan.total_atoms() * 3,
  };
  return gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
             evaluation.plan, positions.data(), kGeometryGeneration, evaluation.matrix.data(),
             evaluation.matrix.size(), evaluation.workspace, evaluation.cache,
             error) == GPUXTB_STATUS_SUCCESS;
}

bool evaluate_energy(const ES2Plan& plan, const ES2GeometryCache& cache,
                     const std::vector<double>& charges, const ES2Workspace& workspace,
                     double& energy, std::string& error) {
  std::array<double, 1> result{};
  if (plan.batch_size() != 1 ||
      gpuxtb::detail::gfn2::add_es2_energy_cpu(plan, cache, charges.data(), result.data(),
                                               workspace, error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  energy = result[0];
  return true;
}

int test_tblite_component_oracle() {
  const std::vector<std::int64_t> offsets{0, 3};
  const std::vector<std::int32_t> atomic_numbers{8, 1, 1};
  const std::vector<double> positions{
      0.0, 0.0, 0.0, 1.43233673, 0.0, 1.10715266, -1.43233673, 0.0, 1.10715266,
  };
  Evaluation evaluation;
  std::string error;
  CHECK(make_evaluation(offsets, atomic_numbers, positions, evaluation, error));
  CHECK(error.empty());
  CHECK(evaluation.plan.total_shells() == 4);

  /*
   * Component oracle from tblite's LGPL-3.0-or-later
   * coulomb/charge/effective.f90 at revision e9abc395: arithmetic_average,
   * get_amat_0d, and gexp=2. Shell parameters are from gpuxtb's separately
   * pinned fa8a441 table. Layout is O(2s), O(2p), H(1s), H(1s), with
   * coordinates in bohr and Gamma in Hartree.
   */
  constexpr std::array<double, 16> expected_matrix{
      0.45189600000000002, 0.48572086749600007, 0.33873660501630865, 0.33873660501630865,
      0.48572086749600007, 0.51954573499200007, 0.35468306344545159, 0.35468306344545159,
      0.33873660501630865, 0.35468306344545159, 0.40577099999999999, 0.26462955009975003,
      0.33873660501630865, 0.35468306344545159, 0.26462955009975003, 0.40577099999999999,
  };
  for (std::size_t index = 0; index < expected_matrix.size(); ++index) {
    CHECK(near(evaluation.matrix[index], expected_matrix[index], 4.0e-16));
  }

  const std::vector<double> charges{
      0.26189717923223715,
      -0.8260775955268945,
      0.23530677010797196,
      0.3288736461866886,
  };
  std::vector<double> potential(4);
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
            evaluation.plan, evaluation.cache, charges.data(), potential.data(),
            evaluation.workspace, error) == GPUXTB_STATUS_SUCCESS);
  constexpr std::array<double, 4> expected_potential{
      -0.09178427977966104,
      -0.10187092804968659,
      -0.021771222425126274,
      -0.0085650578727391874,
  };
  for (std::size_t shell = 0; shell < expected_potential.size(); ++shell) {
    CHECK(near(potential[shell], expected_potential[shell], 5.0e-16));
  }
  std::array<double, 1> energy{};
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(evaluation.plan, evaluation.cache, charges.data(),
                                                 energy.data(), evaluation.workspace,
                                                 error) == GPUXTB_STATUS_SUCCESS);
  CHECK(near(energy[0], 0.026087754741328118, 5.0e-16));
  energy[0] = 0.7;
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(evaluation.plan, evaluation.cache, charges.data(),
                                                 energy.data(), evaluation.workspace,
                                                 error) == GPUXTB_STATUS_SUCCESS);
  CHECK(near(energy[0], 0.7260877547413281, 8.0e-16));
  return 0;
}

int test_long_range_and_heavy_element() {
  {
    const std::vector<std::int64_t> offsets{0, 2};
    const std::vector<std::int32_t> atomic_numbers{1, 1};
    const std::vector<double> positions{0.0, 0.0, 0.0, 1.0e12, 0.0, 0.0};
    Evaluation evaluation;
    std::string error;
    CHECK(make_evaluation(offsets, atomic_numbers, positions, evaluation, error));
    CHECK(relative_near(evaluation.matrix[1], 1.0e-12, 2.0e-16));
    CHECK(evaluation.matrix[1] == evaluation.matrix[2]);
  }

  {
    const std::vector<std::int64_t> offsets{0, 2};
    const std::vector<std::int32_t> atomic_numbers{79, 1};
    const std::vector<double> positions{0.2, -0.3, 0.7, 2.8, 1.1, -0.4};
    Evaluation evaluation;
    std::string error;
    CHECK(make_evaluation(offsets, atomic_numbers, positions, evaluation, error));
    CHECK(evaluation.plan.total_shells() == 4);
    CHECK(evaluation.plan.shell_hardness()[0] == 0.46595694716219999);
    CHECK(evaluation.plan.shell_hardness()[1] == 0.43899700000000003);
    CHECK(evaluation.plan.shell_hardness()[2] == 0.17559880000000003);
    CHECK(near(evaluation.matrix[1],
               0.5 * (evaluation.plan.shell_hardness()[0] + evaluation.plan.shell_hardness()[1]),
               0.0));
    for (double value : evaluation.matrix) {
      CHECK(value > 0.0 && std::isfinite(value));
    }
  }

  return 0;
}

int test_energy_potential_charge_derivative() {
  const std::vector<std::int64_t> offsets{0, 4};
  const std::vector<std::int32_t> atomic_numbers{6, 8, 14, 1};
  const std::vector<double> positions{
      -0.4, 0.2, 1.1, 1.3, -0.7, 0.5, 2.0, 0.3, -1.2, 2.8, -0.9, 0.4,
  };
  Evaluation evaluation;
  std::string error;
  CHECK(make_evaluation(offsets, atomic_numbers, positions, evaluation, error));
  std::vector<double> charges(static_cast<std::size_t>(evaluation.plan.total_shells()));
  for (std::size_t shell = 0; shell < charges.size(); ++shell) {
    charges[shell] = 0.31 * std::sin(0.73 * static_cast<double>(shell + 1u)) - 0.12;
  }
  std::vector<double> potential(charges.size());
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
            evaluation.plan, evaluation.cache, charges.data(), potential.data(),
            evaluation.workspace, error) == GPUXTB_STATUS_SUCCESS);
  double energy = 0.0;
  CHECK(evaluate_energy(evaluation.plan, evaluation.cache, charges, evaluation.workspace, energy,
                        error));
  double charge_dot_potential = 0.0;
  for (std::size_t shell = 0; shell < charges.size(); ++shell) {
    charge_dot_potential += charges[shell] * potential[shell];
  }
  CHECK(near(energy, 0.5 * charge_dot_potential, 3.0e-16));

  constexpr double step = 1.0e-6;
  for (std::size_t shell = 0; shell < charges.size(); ++shell) {
    charges[shell] += step;
    double right = 0.0;
    CHECK(evaluate_energy(evaluation.plan, evaluation.cache, charges, evaluation.workspace, right,
                          error));
    charges[shell] -= 2.0 * step;
    double left = 0.0;
    CHECK(evaluate_energy(evaluation.plan, evaluation.cache, charges, evaluation.workspace, left,
                          error));
    charges[shell] += step;
    CHECK(near((right - left) / (2.0 * step), potential[shell], 4.0e-11));
  }
  return 0;
}

int test_coordinate_vjp() {
  const std::vector<std::int64_t> offsets{0, 4};
  const std::vector<std::int32_t> atomic_numbers{6, 8, 1, 14};
  std::vector<double> positions{
      -0.9, 0.2, 0.7, 1.1, -0.8, 0.4, 0.3, 1.7, -0.5, 2.4, 0.6, 1.3,
  };
  Evaluation evaluation;
  std::string error;
  CHECK(make_evaluation(offsets, atomic_numbers, positions, evaluation, error));
  std::vector<double> charges(static_cast<std::size_t>(evaluation.plan.total_shells()));
  for (std::size_t shell = 0; shell < charges.size(); ++shell) {
    charges[shell] = 0.27 * std::cos(0.61 * static_cast<double>(shell + 2u)) + 0.03;
  }
  std::vector<double> seeds(positions.size());
  for (std::size_t coordinate = 0; coordinate < seeds.size(); ++coordinate) {
    seeds[coordinate] = 0.002 * static_cast<double>(coordinate + 1u);
  }
  std::vector<double> gradients = seeds;
  CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(
            evaluation.plan, evaluation.cache, positions.data(), kGeometryGeneration,
            charges.data(), gradients.data(), evaluation.workspace,
            error) == GPUXTB_STATUS_SUCCESS);

  constexpr double step = 2.0e-5;
  for (std::size_t coordinate = 0; coordinate < positions.size(); ++coordinate) {
    const std::uint64_t right_generation = 1000u + 2u * static_cast<std::uint64_t>(coordinate);
    positions[coordinate] += step;
    CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
              evaluation.plan, positions.data(), right_generation, evaluation.matrix.data(),
              evaluation.matrix.size(), evaluation.workspace, evaluation.cache,
              error) == GPUXTB_STATUS_SUCCESS);
    double right = 0.0;
    CHECK(evaluate_energy(evaluation.plan, evaluation.cache, charges, evaluation.workspace, right,
                          error));

    positions[coordinate] -= 2.0 * step;
    CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
              evaluation.plan, positions.data(), right_generation + 1u, evaluation.matrix.data(),
              evaluation.matrix.size(), evaluation.workspace, evaluation.cache,
              error) == GPUXTB_STATUS_SUCCESS);
    double left = 0.0;
    CHECK(evaluate_energy(evaluation.plan, evaluation.cache, charges, evaluation.workspace, left,
                          error));
    positions[coordinate] += step;
    CHECK(near((right - left) / (2.0 * step), gradients[coordinate] - seeds[coordinate], 2.0e-10));
  }

  for (std::size_t axis = 0; axis < 3u; ++axis) {
    double sum = 0.0;
    for (std::size_t atom = 0; atom < atomic_numbers.size(); ++atom) {
      sum += gradients[atom * 3u + axis] - seeds[atom * 3u + axis];
    }
    CHECK(near(sum, 0.0, 2.0e-17));
  }
  return 0;
}

int test_ragged_matches_sequential() {
  const std::vector<std::int64_t> offsets{0, 2, 2, 5, 6};
  const std::vector<std::int32_t> atomic_numbers{1, 8, 6, 1, 1, 79};
  const std::vector<double> positions{
      0.0, 0.0, 0.0, 1.4, 0.2, -0.3, -0.7, 0.6, 0.1, 1.1, -0.9, 0.8, 2.0, 0.4, -1.2, 3.2, -0.8, 0.5,
  };
  Evaluation batch;
  std::string error;
  CHECK(make_evaluation(offsets, atomic_numbers, positions, batch, error));
  std::vector<double> charges(static_cast<std::size_t>(batch.plan.total_shells()));
  for (std::size_t shell = 0; shell < charges.size(); ++shell) {
    charges[shell] = 0.07 * static_cast<double>(shell + 1u) - 0.31;
  }
  std::vector<double> batch_potential(charges.size());
  std::vector<double> batch_energy(4);
  std::vector<double> batch_gradient(positions.size());
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(batch.plan, batch.cache, charges.data(),
                                                         batch_potential.data(), batch.workspace,
                                                         error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(batch.plan, batch.cache, charges.data(),
                                                 batch_energy.data(), batch.workspace,
                                                 error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(
            batch.plan, batch.cache, positions.data(), kGeometryGeneration, charges.data(),
            batch_gradient.data(), batch.workspace, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(batch_energy[1] == 0.0);

  /* Serial one-system workers over the packed batch reproduce the batch API. */
  for (std::int64_t system = 0; system < batch.plan.batch_size(); ++system) {
    double system_energy = 0.0;
    CHECK(gpuxtb::detail::gfn2::add_es2_energy_system_cpu(batch.plan, batch.cache, system,
                                                          charges.data(), system_energy,
                                                          error) == GPUXTB_STATUS_SUCCESS);
    CHECK(system_energy == batch_energy[static_cast<std::size_t>(system)]);
  }

  for (std::size_t batch_index : {0u, 2u, 3u}) {
    const std::int64_t atom_begin = offsets[batch_index];
    const std::int64_t atom_end = offsets[batch_index + 1u];
    const std::size_t atom_count = static_cast<std::size_t>(atom_end - atom_begin);
    const std::vector<std::int64_t> sequential_offsets{0, static_cast<std::int64_t>(atom_count)};
    const std::vector<std::int32_t> sequential_numbers(atomic_numbers.begin() + atom_begin,
                                                       atomic_numbers.begin() + atom_end);
    const std::vector<double> sequential_positions(positions.begin() + atom_begin * 3,
                                                   positions.begin() + atom_end * 3);
    Evaluation sequential;
    CHECK(make_evaluation(sequential_offsets, sequential_numbers, sequential_positions, sequential,
                          error));

    const std::int64_t shell_begin = batch.plan.batch_shell_offsets()[batch_index];
    const std::int64_t shell_end = batch.plan.batch_shell_offsets()[batch_index + 1u];
    const std::vector<double> sequential_charges(charges.begin() + shell_begin,
                                                 charges.begin() + shell_end);
    std::vector<double> sequential_potential(sequential_charges.size());
    std::array<double, 1> sequential_energy{};
    std::vector<double> sequential_gradient(sequential_positions.size());
    CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
              sequential.plan, sequential.cache, sequential_charges.data(),
              sequential_potential.data(), sequential.workspace, error) == GPUXTB_STATUS_SUCCESS);
    CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(
              sequential.plan, sequential.cache, sequential_charges.data(),
              sequential_energy.data(), sequential.workspace, error) == GPUXTB_STATUS_SUCCESS);
    CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(
              sequential.plan, sequential.cache, sequential_positions.data(), kGeometryGeneration,
              sequential_charges.data(), sequential_gradient.data(), sequential.workspace,
              error) == GPUXTB_STATUS_SUCCESS);

    const std::int64_t matrix_begin = batch.plan.matrix_offsets()[batch_index];
    const std::int64_t matrix_end = batch.plan.matrix_offsets()[batch_index + 1u];
    CHECK(std::equal(batch.matrix.begin() + matrix_begin, batch.matrix.begin() + matrix_end,
                     sequential.matrix.begin()));
    for (std::size_t shell = 0; shell < sequential_potential.size(); ++shell) {
      CHECK(batch_potential[static_cast<std::size_t>(shell_begin) + shell] ==
            sequential_potential[shell]);
    }
    CHECK(batch_energy[batch_index] == sequential_energy[0]);
    for (std::size_t coordinate = 0; coordinate < sequential_gradient.size(); ++coordinate) {
      CHECK(batch_gradient[static_cast<std::size_t>(atom_begin) * 3u + coordinate] ==
            sequential_gradient[coordinate]);
    }
  }
  return 0;
}

int test_system_energy_failure_isolation_and_binding() {
  const std::vector<std::int64_t> offsets{0, 1, 3, 4};
  const std::vector<std::int32_t> atomic_numbers{1, 8, 1, 6};
  const std::vector<double> positions{
      -0.2, 0.1, 0.4, 1.0, -0.7, 0.3, 2.1, 0.2, -0.5, 4.0, 0.6, -0.1,
  };
  Evaluation evaluation;
  std::string error;
  CHECK(make_evaluation(offsets, atomic_numbers, positions, evaluation, error));
  const auto alias_at = [](const void* pointer) {
    return reinterpret_cast<double*>(reinterpret_cast<std::uintptr_t>(pointer));
  };
  std::vector<double> charges(static_cast<std::size_t>(evaluation.plan.total_shells()));
  for (std::size_t shell = 0; shell < charges.size(); ++shell) {
    charges[shell] = -0.23 + 0.09 * static_cast<double>(shell);
  }

  constexpr std::int64_t target = 1;
  const std::size_t target_index = static_cast<std::size_t>(target);
  const std::int64_t target_shell = evaluation.plan.batch_shell_offsets()[target_index];
  const std::int64_t peer_shell = evaluation.plan.batch_shell_offsets()[target_index + 1u];
  const std::int64_t target_matrix = evaluation.plan.matrix_offsets()[target_index];
  const std::int64_t peer_matrix = evaluation.plan.matrix_offsets()[target_index + 1u];
  double expected = 0.375;
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_system_cpu(evaluation.plan, evaluation.cache, target,
                                                        charges.data(), expected,
                                                        error) == GPUXTB_STATUS_SUCCESS);

  /* Numerical poison in another member is deliberately invisible. */
  const double saved_peer_charge = charges[static_cast<std::size_t>(peer_shell)];
  charges[static_cast<std::size_t>(peer_shell)] = std::numeric_limits<double>::quiet_NaN();
  double isolated = 0.375;
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_system_cpu(evaluation.plan, evaluation.cache, target,
                                                        charges.data(), isolated,
                                                        error) == GPUXTB_STATUS_SUCCESS);
  CHECK(isolated == expected);
  charges[static_cast<std::size_t>(peer_shell)] = saved_peer_charge;

  const double saved_peer_matrix = evaluation.matrix[static_cast<std::size_t>(peer_matrix)];
  evaluation.matrix[static_cast<std::size_t>(peer_matrix)] =
      std::numeric_limits<double>::quiet_NaN();
  isolated = 0.375;
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_system_cpu(evaluation.plan, evaluation.cache, target,
                                                        charges.data(), isolated,
                                                        error) == GPUXTB_STATUS_SUCCESS);
  CHECK(isolated == expected);
  evaluation.matrix[static_cast<std::size_t>(peer_matrix)] = saved_peer_matrix;

  const double saved_target_charge = charges[static_cast<std::size_t>(target_shell)];
  charges[static_cast<std::size_t>(target_shell)] = std::numeric_limits<double>::infinity();
  double unchanged = -2.25;
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_system_cpu(evaluation.plan, evaluation.cache, target,
                                                        charges.data(), unchanged,
                                                        error) == GPUXTB_STATUS_INTERNAL_ERROR);
  CHECK(unchanged == -2.25);
  charges[static_cast<std::size_t>(target_shell)] = saved_target_charge;

  const double saved_target_matrix = evaluation.matrix[static_cast<std::size_t>(target_matrix)];
  evaluation.matrix[static_cast<std::size_t>(target_matrix)] =
      std::numeric_limits<double>::quiet_NaN();
  unchanged = -1.75;
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_system_cpu(evaluation.plan, evaluation.cache, target,
                                                        charges.data(), unchanged,
                                                        error) == GPUXTB_STATUS_INTERNAL_ERROR);
  CHECK(unchanged == -1.75);
  evaluation.matrix[static_cast<std::size_t>(target_matrix)] = saved_target_matrix;

  unchanged = 4.5;
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_system_cpu(evaluation.plan, evaluation.cache, -1,
                                                        charges.data(), unchanged,
                                                        error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(unchanged == 4.5);
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_system_cpu(
            evaluation.plan, evaluation.cache, evaluation.plan.batch_size(), charges.data(),
            unchanged, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(unchanged == 4.5);

  /* A same-shape cache from another sealed plan does not establish provenance. */
  Evaluation foreign;
  CHECK(make_evaluation(offsets, atomic_numbers, positions, foreign, error));
  CHECK(foreign.plan.identity() != evaluation.plan.identity());
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_system_cpu(evaluation.plan, foreign.cache, target,
                                                        charges.data(), unchanged,
                                                        error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(unchanged == 4.5);

  const std::vector<double> saved_charges = charges;
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_system_cpu(evaluation.plan, evaluation.cache, target,
                                                        charges.data(), charges[0],
                                                        error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(charges == saved_charges);

  const std::vector<double> saved_matrix = evaluation.matrix;
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_system_cpu(evaluation.plan, evaluation.cache, target,
                                                        charges.data(), evaluation.matrix[0],
                                                        error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(evaluation.matrix == saved_matrix);

  const double saved_hardness = evaluation.plan.shell_hardness()[0];
  double& plan_alias = const_cast<double&>(evaluation.plan.shell_hardness()[0]);
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_system_cpu(evaluation.plan, evaluation.cache, target,
                                                        charges.data(), plan_alias,
                                                        error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(evaluation.plan.shell_hardness()[0] == saved_hardness);

  const ES2GeometryCache saved_cache = evaluation.cache;
  double& cache_alias = *alias_at(&evaluation.cache);
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_system_cpu(evaluation.plan, evaluation.cache, target,
                                                        charges.data(), cache_alias,
                                                        error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(same_cache_descriptor(evaluation.cache, saved_cache));

  unchanged = 3.25;
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_system_cpu(
            evaluation.plan, evaluation.cache, target, evaluation.plan.shell_hardness().data(),
            unchanged, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(unchanged == 3.25);
  return 0;
}

int test_system_potential_matches_batch_and_failure_isolation() {
  const std::vector<std::int64_t> offsets{0, 1, 3, 4};
  const std::vector<std::int32_t> atomic_numbers{1, 8, 1, 6};
  const std::vector<double> positions{
      -0.2, 0.1, 0.4, 1.0, -0.7, 0.3, 2.1, 0.2, -0.5, 4.0, 0.6, -0.1,
  };
  Evaluation evaluation;
  std::string error;
  CHECK(make_evaluation(offsets, atomic_numbers, positions, evaluation, error));
  std::vector<double> charges(static_cast<std::size_t>(evaluation.plan.total_shells()));
  for (std::size_t shell = 0; shell < charges.size(); ++shell) {
    charges[shell] = -0.23 + 0.09 * static_cast<double>(shell);
  }

  const std::int64_t batch = evaluation.plan.batch_size();
  std::vector<double> batch_potential(charges.size());
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
            evaluation.plan, evaluation.cache, charges.data(), batch_potential.data(),
            evaluation.workspace, error) == GPUXTB_STATUS_SUCCESS);

  /* Per-system potential reproduces the batch API's target slice. */
  std::vector<double> system_potential(charges.size(), 0.0);
  for (std::int64_t system = 0; system < batch; ++system) {
    const std::int64_t shell_begin = evaluation.plan.batch_shell_offsets()[system];
    const std::int64_t shell_end = evaluation.plan.batch_shell_offsets()[system + 1u];
    CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_system_cpu(
              evaluation.plan, evaluation.cache, system, charges.data(), system_potential.data(),
              error) == GPUXTB_STATUS_SUCCESS);
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      CHECK(system_potential[static_cast<std::size_t>(shell)] ==
            batch_potential[static_cast<std::size_t>(shell)]);
    }
  }

  /* A poisoned peer charge or matrix must not affect the target system. */
  constexpr std::int64_t target = 1;
  const std::size_t target_index = static_cast<std::size_t>(target);
  const std::int64_t peer_shell = evaluation.plan.batch_shell_offsets()[target_index + 1u];
  const std::int64_t target_matrix = evaluation.plan.matrix_offsets()[target_index];
  const std::int64_t peer_matrix = evaluation.plan.matrix_offsets()[target_index + 1u];
  const std::vector<double> expected(batch_potential);

  const double saved_peer_charge = charges[static_cast<std::size_t>(peer_shell)];
  charges[static_cast<std::size_t>(peer_shell)] = std::numeric_limits<double>::quiet_NaN();
  std::fill(system_potential.begin(), system_potential.end(), 0.0);
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_system_cpu(
            evaluation.plan, evaluation.cache, target, charges.data(), system_potential.data(),
            error) == GPUXTB_STATUS_SUCCESS);
  {
    const std::int64_t shell_begin = evaluation.plan.batch_shell_offsets()[target_index];
    const std::int64_t shell_end = evaluation.plan.batch_shell_offsets()[target_index + 1u];
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      CHECK(system_potential[static_cast<std::size_t>(shell)] ==
            expected[static_cast<std::size_t>(shell)]);
    }
  }
  charges[static_cast<std::size_t>(peer_shell)] = saved_peer_charge;

  const double saved_peer_matrix = evaluation.matrix[static_cast<std::size_t>(peer_matrix)];
  evaluation.matrix[static_cast<std::size_t>(peer_matrix)] =
      std::numeric_limits<double>::quiet_NaN();
  std::fill(system_potential.begin(), system_potential.end(), 0.0);
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_system_cpu(
            evaluation.plan, evaluation.cache, target, charges.data(), system_potential.data(),
            error) == GPUXTB_STATUS_SUCCESS);
  {
    const std::int64_t shell_begin = evaluation.plan.batch_shell_offsets()[target_index];
    const std::int64_t shell_end = evaluation.plan.batch_shell_offsets()[target_index + 1u];
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      CHECK(system_potential[static_cast<std::size_t>(shell)] ==
            expected[static_cast<std::size_t>(shell)]);
    }
  }
  evaluation.matrix[static_cast<std::size_t>(peer_matrix)] = saved_peer_matrix;

  /* Target numerical poison is a target-only failure. */
  const double saved_target_charge =
      charges[static_cast<std::size_t>(evaluation.plan.batch_shell_offsets()[target_index])];
  charges[static_cast<std::size_t>(evaluation.plan.batch_shell_offsets()[target_index])] =
      std::numeric_limits<double>::quiet_NaN();
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_system_cpu(
            evaluation.plan, evaluation.cache, target, charges.data(), system_potential.data(),
            error) == GPUXTB_STATUS_INTERNAL_ERROR);
  charges[static_cast<std::size_t>(evaluation.plan.batch_shell_offsets()[target_index])] =
      saved_target_charge;

  const double saved_target_matrix = evaluation.matrix[static_cast<std::size_t>(target_matrix)];
  evaluation.matrix[static_cast<std::size_t>(target_matrix)] =
      std::numeric_limits<double>::quiet_NaN();
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_system_cpu(
            evaluation.plan, evaluation.cache, target, charges.data(), system_potential.data(),
            error) == GPUXTB_STATUS_INTERNAL_ERROR);
  evaluation.matrix[static_cast<std::size_t>(target_matrix)] = saved_target_matrix;

  /* Out-of-range system and foreign cache are structural failures. */
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_system_cpu(
            evaluation.plan, evaluation.cache, -1, charges.data(), system_potential.data(),
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_system_cpu(
            evaluation.plan, evaluation.cache, batch, charges.data(), system_potential.data(),
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  Evaluation foreign;
  CHECK(make_evaluation(offsets, atomic_numbers, positions, foreign, error));
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_system_cpu(
            evaluation.plan, foreign.cache, target, charges.data(), system_potential.data(),
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);

  /* Output aliasing plan storage and a NULL output are rejected. */
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_system_cpu(
            evaluation.plan, evaluation.cache, target, evaluation.plan.shell_hardness().data(),
            system_potential.data(), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_system_cpu(
            evaluation.plan, evaluation.cache, target, charges.data(), nullptr, error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);

  /* The one-system potential allocates nothing and needs no scratch. */
  const std::size_t before = allocation_test::count.load(std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const gpuxtb_status_t status = gpuxtb::detail::gfn2::evaluate_es2_potential_system_cpu(
      evaluation.plan, evaluation.cache, target, charges.data(), system_potential.data(), error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  CHECK(status == GPUXTB_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == before);
  return 0;
}

int test_cache_plan_identity() {
  const std::vector<std::int64_t> pair_offsets{0, 2};
  const std::vector<std::int32_t> pair_numbers{1, 1};
  const std::vector<double> pair_positions{0.0, 0.0, 0.0, 1.4, 0.0, 0.0};
  Evaluation pair;
  std::string error;
  CHECK(make_evaluation(pair_offsets, pair_numbers, pair_positions, pair, error));

  const std::vector<std::int64_t> isolated_offsets{0, 1, 2, 3, 4};
  const std::vector<std::int32_t> isolated_numbers{1, 1, 1, 1};
  const std::vector<double> isolated_positions{
      0.0, 0.0, 0.0, 2.0, 0.1, 0.0, 4.0, -0.2, 0.3, 6.0, 0.4, -0.1,
  };
  Evaluation isolated;
  CHECK(make_evaluation(isolated_offsets, isolated_numbers, isolated_positions, isolated, error));
  CHECK(pair.plan.total_matrix_elements() == isolated.plan.total_matrix_elements());
  CHECK(pair.plan.identity() != isolated.plan.identity());

  /* Copies of one immutable plan deliberately remain cache-compatible. */
  ES2Plan compatible_copy;
  const std::size_t allocations_before = allocation_test::count.load(std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  compatible_copy = pair.plan;
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  const std::size_t allocations_after = allocation_test::count.load(std::memory_order_relaxed);
  CHECK(allocations_after == allocations_before);
  CHECK(compatible_copy.identity() == pair.plan.identity());
  CHECK(compatible_copy.atom_offsets().data() == pair.plan.atom_offsets().data());
  CHECK(compatible_copy.batch_shell_offsets().data() == pair.plan.batch_shell_offsets().data());
  CHECK(compatible_copy.atom_shell_offsets().data() == pair.plan.atom_shell_offsets().data());
  CHECK(compatible_copy.matrix_offsets().data() == pair.plan.matrix_offsets().data());
  CHECK(compatible_copy.shell_to_atom().data() == pair.plan.shell_to_atom().data());
  CHECK(compatible_copy.shell_hardness().data() == pair.plan.shell_hardness().data());
  const std::array<double, 2> pair_charges{0.2, -0.3};
  std::array<double, 2> pair_potential{};
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
            compatible_copy, pair.cache, pair_charges.data(), pair_potential.data(), pair.workspace,
            error) == GPUXTB_STATUS_SUCCESS);

  const std::array<double, 4> isolated_charges{0.1, -0.2, 0.3, -0.4};
  std::array<double, 4> potential{9.0, 9.0, 9.0, 9.0};
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
            isolated.plan, pair.cache, isolated_charges.data(), potential.data(),
            isolated.workspace, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(potential.begin(), potential.end(), [](double value) { return value == 9.0; }));

  std::array<double, 4> energies{8.0, 8.0, 8.0, 8.0};
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(isolated.plan, pair.cache, isolated_charges.data(),
                                                 energies.data(), isolated.workspace,
                                                 error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(energies.begin(), energies.end(), [](double value) { return value == 8.0; }));

  std::vector<double> gradients(isolated_positions.size(), 7.0);
  CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(
            isolated.plan, pair.cache, isolated_positions.data(), kGeometryGeneration,
            isolated_charges.data(), gradients.data(), isolated.workspace,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(gradients.begin(), gradients.end(), [](double value) { return value == 7.0; }));
  return 0;
}

int test_default_and_moved_from_plans_are_rejected_atomically() {
  const std::array<double, 6> positions{0.0, 0.0, 0.0, 1.4, 0.0, 0.0};
  const std::array<double, 2> charges{0.2, -0.3};
  std::array<double, 4> matrix{6.0, 6.0, 6.0, 6.0};
  std::array<double, 4> matrix_scratch{};
  std::array<double, 2> shell_scratch{};
  std::array<double, 1> batch_scratch{};
  std::array<double, 6> gradient_scratch{};
  const ES2Workspace workspace{matrix_scratch.data(), 4, shell_scratch.data(),    2,
                               batch_scratch.data(),  1, gradient_scratch.data(), 6};
  ES2GeometryCache cache;
  std::string error;

  const auto rejects_without_publication = [&](const ES2Plan& plan) {
    matrix.fill(6.0);
    cache = {};
    if (gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
            plan, positions.data(), kGeometryGeneration, matrix.data(), matrix.size(), workspace,
            cache, error) != GPUXTB_STATUS_INVALID_ARGUMENT ||
        !std::all_of(matrix.begin(), matrix.end(), [](double value) { return value == 6.0; }) ||
        cache.coulomb_matrix != nullptr || cache.matrix_elements != 0 ||
        cache.geometry_generation != 0u || cache.plan_identity != nullptr) {
      return false;
    }

    std::array<double, 2> potential{7.0, 7.0};
    if (gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(plan, cache, charges.data(),
                                                         potential.data(), workspace,
                                                         error) != GPUXTB_STATUS_INVALID_ARGUMENT ||
        potential[0] != 7.0 || potential[1] != 7.0) {
      return false;
    }

    std::array<double, 1> energy{8.0};
    if (gpuxtb::detail::gfn2::add_es2_energy_cpu(plan, cache, charges.data(), energy.data(),
                                                 workspace,
                                                 error) != GPUXTB_STATUS_INVALID_ARGUMENT ||
        energy[0] != 8.0) {
      return false;
    }

    std::array<double, 6> gradient{9.0, 9.0, 9.0, 9.0, 9.0, 9.0};
    return gpuxtb::detail::gfn2::add_es2_gradient_cpu(
               plan, cache, positions.data(), kGeometryGeneration, charges.data(), gradient.data(),
               workspace, error) == GPUXTB_STATUS_INVALID_ARGUMENT &&
           std::all_of(gradient.begin(), gradient.end(), [](double value) { return value == 9.0; });
  };

  const ES2Plan default_plan;
  CHECK(!default_plan.sealed());
  CHECK(default_plan.identity() == nullptr);
  CHECK(default_plan.batch_size() == 0);
  CHECK(default_plan.total_atoms() == 0);
  CHECK(default_plan.total_shells() == 0);
  CHECK(default_plan.total_matrix_elements() == 0);
  CHECK(default_plan.atom_offsets().empty());
  CHECK(default_plan.batch_shell_offsets().empty());
  CHECK(default_plan.atom_shell_offsets().empty());
  CHECK(default_plan.matrix_offsets().empty());
  CHECK(default_plan.shell_to_atom().empty());
  CHECK(default_plan.shell_hardness().empty());
  CHECK(rejects_without_publication(default_plan));

  const std::vector<std::int64_t> offsets{0, 2};
  const std::vector<std::int32_t> atomic_numbers{1, 1};
  const std::vector<double> valid_positions(positions.begin(), positions.end());
  Evaluation evaluation;
  CHECK(make_evaluation(offsets, atomic_numbers, valid_positions, evaluation, error));
  ES2Plan moved_from = evaluation.plan;
  ES2Plan live_plan = std::move(moved_from);
  CHECK(!moved_from.sealed());
  CHECK(moved_from.identity() == nullptr);
  CHECK(rejects_without_publication(moved_from));

  std::array<double, 2> potential{};
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
            live_plan, evaluation.cache, charges.data(), potential.data(), evaluation.workspace,
            error) == GPUXTB_STATUS_SUCCESS);
  return 0;
}

int test_plan_storage_aliases_are_rejected_atomically() {
  const std::vector<std::int64_t> offsets{0, 2};
  const std::vector<std::int32_t> atomic_numbers{1, 1};
  const std::vector<double> positions{0.0, 0.0, 0.0, 1.4, 0.0, 0.0};
  const std::vector<double> charges{0.2, -0.3};
  Evaluation evaluation;
  std::string error;
  CHECK(make_evaluation(offsets, atomic_numbers, positions, evaluation, error));

  const std::vector<std::int64_t> atom_offsets_before = evaluation.plan.atom_offsets();
  const std::vector<double> hardness_before = evaluation.plan.shell_hardness();
  double* const hardness = const_cast<double*>(evaluation.plan.shell_hardness().data());
  double* const hardness_partial = hardness + 1;
  double* const metadata =
      reinterpret_cast<double*>(const_cast<std::int64_t*>(evaluation.plan.atom_offsets().data()));
  CHECK(reinterpret_cast<std::uintptr_t>(metadata) % alignof(double) == 0u);

  const auto plan_is_unchanged = [&]() {
    return evaluation.plan.atom_offsets() == atom_offsets_before &&
           evaluation.plan.shell_hardness() == hardness_before;
  };

  std::vector<double> matrix_output(evaluation.matrix.size(), 6.0);
  ES2GeometryCache unpublished_cache;
  CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
            evaluation.plan, positions.data(), kGeometryGeneration + 1u, hardness,
            evaluation.matrix.size(), evaluation.workspace, unpublished_cache,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(plan_is_unchanged());
  CHECK(unpublished_cache.coulomb_matrix == nullptr && unpublished_cache.plan_identity == nullptr);

  ES2Workspace alias_workspace = evaluation.workspace;
  alias_workspace.matrix_scratch = hardness_partial;
  CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
            evaluation.plan, positions.data(), kGeometryGeneration + 1u, matrix_output.data(),
            matrix_output.size(), alias_workspace, unpublished_cache,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(matrix_output.begin(), matrix_output.end(),
                    [](double value) { return value == 6.0; }));
  CHECK(plan_is_unchanged());

  alias_workspace = evaluation.workspace;
  alias_workspace.matrix_scratch = metadata;
  CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
            evaluation.plan, positions.data(), kGeometryGeneration + 1u, matrix_output.data(),
            matrix_output.size(), alias_workspace, unpublished_cache,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(matrix_output.begin(), matrix_output.end(),
                    [](double value) { return value == 6.0; }));
  CHECK(plan_is_unchanged());

  ES2GeometryCache forged_cache{hardness, evaluation.plan.total_matrix_elements(),
                                kGeometryGeneration, evaluation.plan.identity()};
  const ES2GeometryCache forged_cache_before = forged_cache;
  CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
            evaluation.plan, positions.data(), kGeometryGeneration + 1u, matrix_output.data(),
            matrix_output.size(), evaluation.workspace, forged_cache,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(forged_cache.coulomb_matrix == forged_cache_before.coulomb_matrix);
  CHECK(forged_cache.matrix_elements == forged_cache_before.matrix_elements);
  CHECK(forged_cache.geometry_generation == forged_cache_before.geometry_generation);
  CHECK(forged_cache.plan_identity == forged_cache_before.plan_identity);
  CHECK(plan_is_unchanged());

  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
            evaluation.plan, evaluation.cache, charges.data(), hardness, evaluation.workspace,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(plan_is_unchanged());

  std::array<double, 2> potential{7.0, 7.0};
  alias_workspace = evaluation.workspace;
  alias_workspace.shell_scratch = hardness_partial;
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
            evaluation.plan, evaluation.cache, charges.data(), potential.data(), alias_workspace,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(potential[0] == 7.0 && potential[1] == 7.0);
  CHECK(plan_is_unchanged());

  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
            evaluation.plan, evaluation.cache, charges.data(), metadata, evaluation.workspace,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(plan_is_unchanged());

  potential = {7.0, 7.0};
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
            evaluation.plan, forged_cache, charges.data(), potential.data(), evaluation.workspace,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(potential[0] == 7.0 && potential[1] == 7.0);
  CHECK(plan_is_unchanged());

  CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(evaluation.plan, evaluation.cache, charges.data(),
                                                 metadata, evaluation.workspace,
                                                 error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(plan_is_unchanged());

  std::array<double, 1> energy{8.0};
  alias_workspace = evaluation.workspace;
  alias_workspace.batch_scratch = hardness_partial;
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(evaluation.plan, evaluation.cache, charges.data(),
                                                 energy.data(), alias_workspace,
                                                 error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(energy[0] == 8.0);
  CHECK(plan_is_unchanged());

  CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(evaluation.plan, forged_cache, charges.data(),
                                                 energy.data(), evaluation.workspace,
                                                 error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(energy[0] == 8.0);
  CHECK(plan_is_unchanged());

  CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(evaluation.plan, evaluation.cache,
                                                   positions.data(), kGeometryGeneration,
                                                   charges.data(), hardness, evaluation.workspace,
                                                   error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(plan_is_unchanged());

  std::array<double, 6> gradient{9.0, 9.0, 9.0, 9.0, 9.0, 9.0};
  alias_workspace = evaluation.workspace;
  alias_workspace.gradient_scratch = metadata;
  CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(evaluation.plan, evaluation.cache,
                                                   positions.data(), kGeometryGeneration,
                                                   charges.data(), gradient.data(), alias_workspace,
                                                   error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(gradient.begin(), gradient.end(), [](double value) { return value == 9.0; }));
  CHECK(plan_is_unchanged());

  CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(
            evaluation.plan, forged_cache, positions.data(), kGeometryGeneration, charges.data(),
            gradient.data(), evaluation.workspace, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(gradient.begin(), gradient.end(), [](double value) { return value == 9.0; }));
  CHECK(plan_is_unchanged());
  return 0;
}

int test_opaque_plan_object_aliases_and_active_cache_descriptor() {
  const std::vector<std::int64_t> offsets{0, 2};
  const std::vector<std::int32_t> atomic_numbers{1, 1};
  const std::vector<double> positions{0.0, 0.0, 0.0, 1.4, 0.0, 0.0};
  const std::vector<double> charges{0.2, -0.3};
  Evaluation evaluation;
  std::string error;
  CHECK(make_evaluation(offsets, atomic_numbers, positions, evaluation, error));

  const auto alias_at = [](const void* pointer, std::size_t offset_bytes) {
    return reinterpret_cast<double*>(reinterpret_cast<std::uintptr_t>(pointer) + offset_bytes);
  };
  const std::array<double*, 4> plan_object_aliases{
      alias_at(evaluation.plan.identity(), 0u),
      alias_at(evaluation.plan.identity(), sizeof(double)),
      alias_at(&evaluation.plan, 0u),
      alias_at(&evaluation.plan, sizeof(double)),
  };
  for (double* alias : plan_object_aliases) {
    CHECK(reinterpret_cast<std::uintptr_t>(alias) % alignof(double) == 0u);
  }

  const auto* identity_before = evaluation.plan.identity();
  const std::vector<std::int64_t> atom_offsets_before = evaluation.plan.atom_offsets();
  const std::vector<std::int64_t> shell_offsets_before = evaluation.plan.batch_shell_offsets();
  const std::vector<std::int64_t> atom_shell_offsets_before = evaluation.plan.atom_shell_offsets();
  const std::vector<std::int64_t> matrix_offsets_before = evaluation.plan.matrix_offsets();
  const std::vector<std::int64_t> shell_to_atom_before = evaluation.plan.shell_to_atom();
  const std::vector<double> hardness_before = evaluation.plan.shell_hardness();
  const auto plan_is_unchanged = [&]() {
    return evaluation.plan.sealed() && evaluation.plan.identity() == identity_before &&
           evaluation.plan.batch_size() == 1 && evaluation.plan.total_atoms() == 2 &&
           evaluation.plan.total_shells() == 2 && evaluation.plan.total_matrix_elements() == 4 &&
           evaluation.plan.atom_offsets() == atom_offsets_before &&
           evaluation.plan.batch_shell_offsets() == shell_offsets_before &&
           evaluation.plan.atom_shell_offsets() == atom_shell_offsets_before &&
           evaluation.plan.matrix_offsets() == matrix_offsets_before &&
           evaluation.plan.shell_to_atom() == shell_to_atom_before &&
           evaluation.plan.shell_hardness() == hardness_before;
  };
  const auto cache_is_default = [](const ES2GeometryCache& cache) {
    return cache.coulomb_matrix == nullptr && cache.matrix_elements == 0 &&
           cache.geometry_generation == 0u && cache.plan_identity == nullptr;
  };

  for (double* alias : plan_object_aliases) {
    ES2GeometryCache unpublished_cache;
    CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
              evaluation.plan, positions.data(), kGeometryGeneration + 1u, alias,
              evaluation.matrix.size(), evaluation.workspace, unpublished_cache,
              error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(cache_is_default(unpublished_cache));
    CHECK(plan_is_unchanged());

    CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
              evaluation.plan, evaluation.cache, charges.data(), alias, evaluation.workspace,
              error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(plan_is_unchanged());

    CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(evaluation.plan, evaluation.cache,
                                                   charges.data(), alias, evaluation.workspace,
                                                   error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(plan_is_unchanged());

    CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(evaluation.plan, evaluation.cache,
                                                     positions.data(), kGeometryGeneration,
                                                     charges.data(), alias, evaluation.workspace,
                                                     error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(plan_is_unchanged());
  }

  std::vector<double> matrix_output(evaluation.matrix.size(), 6.0);
  std::array<double, 2> potential{7.0, 7.0};
  std::array<double, 1> energy{8.0};
  std::array<double, 6> gradient{9.0, 9.0, 9.0, 9.0, 9.0, 9.0};
  for (std::size_t alias_index = 0; alias_index < plan_object_aliases.size(); ++alias_index) {
    ES2Workspace alias_workspace = evaluation.workspace;
    alias_workspace.matrix_scratch = plan_object_aliases[alias_index];
    ES2GeometryCache unpublished_cache;
    CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
              evaluation.plan, positions.data(), kGeometryGeneration + 1u, matrix_output.data(),
              matrix_output.size(), alias_workspace, unpublished_cache,
              error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(std::all_of(matrix_output.begin(), matrix_output.end(),
                      [](double value) { return value == 6.0; }));
    CHECK(cache_is_default(unpublished_cache));
    CHECK(plan_is_unchanged());

    alias_workspace = evaluation.workspace;
    alias_workspace.shell_scratch = plan_object_aliases[alias_index];
    CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
              evaluation.plan, evaluation.cache, charges.data(), potential.data(), alias_workspace,
              error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(potential[0] == 7.0 && potential[1] == 7.0);
    CHECK(plan_is_unchanged());

    alias_workspace = evaluation.workspace;
    alias_workspace.batch_scratch = plan_object_aliases[alias_index];
    CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(evaluation.plan, evaluation.cache,
                                                   charges.data(), energy.data(), alias_workspace,
                                                   error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(energy[0] == 8.0);
    CHECK(plan_is_unchanged());

    alias_workspace = evaluation.workspace;
    alias_workspace.gradient_scratch = plan_object_aliases[alias_index];
    CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(
              evaluation.plan, evaluation.cache, positions.data(), kGeometryGeneration,
              charges.data(), gradient.data(), alias_workspace,
              error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(std::all_of(gradient.begin(), gradient.end(), [](double value) { return value == 9.0; }));
    CHECK(plan_is_unchanged());

    const ES2GeometryCache forged_cache{plan_object_aliases[alias_index],
                                        evaluation.plan.total_matrix_elements(),
                                        kGeometryGeneration, evaluation.plan.identity()};
    ES2GeometryCache active_forged_cache = forged_cache;
    const ES2GeometryCache active_forged_before = active_forged_cache;
    CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
              evaluation.plan, positions.data(), kGeometryGeneration + 1u, matrix_output.data(),
              matrix_output.size(), evaluation.workspace, active_forged_cache,
              error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(std::all_of(matrix_output.begin(), matrix_output.end(),
                      [](double value) { return value == 6.0; }));
    CHECK(active_forged_cache.coulomb_matrix == active_forged_before.coulomb_matrix);
    CHECK(active_forged_cache.matrix_elements == active_forged_before.matrix_elements);
    CHECK(active_forged_cache.geometry_generation == active_forged_before.geometry_generation);
    CHECK(active_forged_cache.plan_identity == active_forged_before.plan_identity);
    CHECK(plan_is_unchanged());

    potential = {7.0, 7.0};
    CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
              evaluation.plan, forged_cache, charges.data(), potential.data(), evaluation.workspace,
              error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(potential[0] == 7.0 && potential[1] == 7.0);
    CHECK(plan_is_unchanged());
  }

  /*
   * Even a corrupt count must not suppress active-cache/scratch alias checking.
   * The late coordinate overflow would otherwise occur after onsite scratch
   * entries had already overwritten the active cache backing storage.
   */
  std::fill(evaluation.matrix_scratch.begin(), evaluation.matrix_scratch.end(), 12.0);
  matrix_output.assign(matrix_output.size(), 13.0);
  const std::vector<double> overflowing_positions{
      std::numeric_limits<double>::max(), 0.0, 0.0, -std::numeric_limits<double>::max(), 0.0, 0.0,
  };
  ES2GeometryCache corrupt_active_cache{evaluation.matrix_scratch.data(), 0, kGeometryGeneration,
                                        evaluation.plan.identity()};
  const ES2GeometryCache corrupt_before = corrupt_active_cache;
  CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
            evaluation.plan, overflowing_positions.data(), kGeometryGeneration + 1u,
            matrix_output.data(), matrix_output.size(), evaluation.workspace, corrupt_active_cache,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(evaluation.matrix_scratch.begin(), evaluation.matrix_scratch.end(),
                    [](double value) { return value == 12.0; }));
  CHECK(std::all_of(matrix_output.begin(), matrix_output.end(),
                    [](double value) { return value == 13.0; }));
  CHECK(corrupt_active_cache.coulomb_matrix == corrupt_before.coulomb_matrix);
  CHECK(corrupt_active_cache.matrix_elements == corrupt_before.matrix_elements);
  CHECK(corrupt_active_cache.geometry_generation == corrupt_before.geometry_generation);
  CHECK(corrupt_active_cache.plan_identity == corrupt_before.plan_identity);
  CHECK(plan_is_unchanged());
  return 0;
}

int test_descriptor_object_alias_matrix() {
  const std::vector<std::int64_t> offsets{0, 2};
  const std::vector<std::int32_t> atomic_numbers{1, 1};
  const std::vector<double> positions{0.0, 0.0, 0.0, 1.4, 0.0, 0.0};
  const std::vector<double> charges{0.2, -0.3};
  Evaluation evaluation;
  std::string error;
  CHECK(make_evaluation(offsets, atomic_numbers, positions, evaluation, error));

  const auto alias_at = [](const void* pointer, std::size_t offset_bytes) {
    return reinterpret_cast<double*>(reinterpret_cast<std::uintptr_t>(pointer) + offset_bytes);
  };
  const auto tail_is = [](const auto& envelope, std::byte value) {
    return std::all_of(envelope.tail.begin(), envelope.tail.end(),
                       [value](std::byte actual) { return actual == value; });
  };
  const auto outputs_are_unchanged = [](const std::vector<double>& matrix_output,
                                        const std::array<double, 2>& potential,
                                        const std::array<double, 1>& energy,
                                        const std::array<double, 6>& gradient) {
    return std::all_of(matrix_output.begin(), matrix_output.end(),
                       [](double value) { return value == 6.0; }) &&
           potential[0] == 7.0 && potential[1] == 7.0 && energy[0] == 8.0 &&
           std::all_of(gradient.begin(), gradient.end(), [](double value) { return value == 9.0; });
  };

  const auto* identity_before = evaluation.plan.identity();
  const std::vector<std::int64_t> atom_offsets_before = evaluation.plan.atom_offsets();
  const std::vector<double> hardness_before = evaluation.plan.shell_hardness();
  const ES2GeometryCache cache_before = evaluation.cache;
  const ES2Workspace workspace_before = evaluation.workspace;

  /*
   * Input validation must reject protected storage before it reads the number
   * of doubles implied by the plan. Several of these objects are intentionally
   * smaller than a full coordinate buffer, so sanitizer coverage also proves
   * that descriptor overlap is checked before numeric value validation.
   */
  const std::array<double*, 12> input_aliases{
      alias_at(evaluation.plan.identity(), 0u),
      alias_at(evaluation.plan.identity(), sizeof(double)),
      alias_at(&evaluation.plan, 0u),
      alias_at(&evaluation.plan, sizeof(double)),
      reinterpret_cast<double*>(const_cast<std::int64_t*>(evaluation.plan.atom_offsets().data())),
      const_cast<double*>(evaluation.plan.shell_hardness().data()),
      alias_at(&evaluation.cache, 0u),
      alias_at(&evaluation.cache, sizeof(double)),
      alias_at(&evaluation.workspace, 0u),
      alias_at(&evaluation.workspace, sizeof(double)),
      alias_at(&evaluation.workspace, 2u * sizeof(double)),
      alias_at(&evaluation.workspace, 3u * sizeof(double)),
  };
  for (double* alias : input_aliases) {
    CHECK(reinterpret_cast<std::uintptr_t>(alias) % alignof(double) == 0u);
    std::vector<double> matrix_output(evaluation.matrix.size(), 6.0);
    std::array<double, 2> potential{7.0, 7.0};
    std::array<double, 1> energy{8.0};
    std::array<double, 6> gradient{9.0, 9.0, 9.0, 9.0, 9.0, 9.0};
    const std::vector<double> matrix_scratch_before = evaluation.matrix_scratch;
    const std::vector<double> shell_scratch_before = evaluation.shell_scratch;
    const std::vector<double> batch_scratch_before = evaluation.batch_scratch;
    const std::vector<double> gradient_scratch_before = evaluation.gradient_scratch;

    CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
              evaluation.plan, alias, kGeometryGeneration + 1u, matrix_output.data(),
              matrix_output.size(), evaluation.workspace, evaluation.cache,
              error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
              evaluation.plan, evaluation.cache, alias, potential.data(), evaluation.workspace,
              error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(evaluation.plan, evaluation.cache, alias,
                                                   energy.data(), evaluation.workspace,
                                                   error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(
              evaluation.plan, evaluation.cache, alias, kGeometryGeneration, charges.data(),
              gradient.data(), evaluation.workspace, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(
              evaluation.plan, evaluation.cache, positions.data(), kGeometryGeneration, alias,
              gradient.data(), evaluation.workspace, error) == GPUXTB_STATUS_INVALID_ARGUMENT);

    CHECK(outputs_are_unchanged(matrix_output, potential, energy, gradient));
    CHECK(evaluation.matrix_scratch == matrix_scratch_before);
    CHECK(evaluation.shell_scratch == shell_scratch_before);
    CHECK(evaluation.batch_scratch == batch_scratch_before);
    CHECK(evaluation.gradient_scratch == gradient_scratch_before);
    CHECK(same_cache_descriptor(evaluation.cache, cache_before));
    CHECK(same_workspace_descriptor(evaluation.workspace, workspace_before));
    CHECK(evaluation.plan.identity() == identity_before);
    CHECK(evaluation.plan.atom_offsets() == atom_offsets_before);
    CHECK(evaluation.plan.shell_hardness() == hardness_before);
  }

  constexpr std::byte kTailSentinel{0x6d};
  for (const std::size_t alias_offset : {std::size_t{0}, sizeof(double)}) {
    /* Output and active scratch aliases into the cache descriptor. */
    DescriptorEnvelope<ES2GeometryCache> cache_envelope;
    cache_envelope.descriptor = evaluation.cache;
    cache_envelope.tail.fill(kTailSentinel);
    double* cache_alias = alias_at(&cache_envelope.descriptor, alias_offset);
    ES2Workspace local_workspace = evaluation.workspace;
    std::vector<double> matrix_output(evaluation.matrix.size(), 6.0);
    std::array<double, 2> potential{7.0, 7.0};
    std::array<double, 1> energy{8.0};
    std::array<double, 6> gradient{9.0, 9.0, 9.0, 9.0, 9.0, 9.0};
    ES2GeometryCache descriptor_before = cache_envelope.descriptor;

    CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
              evaluation.plan, positions.data(), kGeometryGeneration + 1u, cache_alias,
              evaluation.matrix.size(), local_workspace, cache_envelope.descriptor,
              error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
              evaluation.plan, cache_envelope.descriptor, charges.data(), cache_alias,
              local_workspace, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(evaluation.plan, cache_envelope.descriptor,
                                                   charges.data(), cache_alias, local_workspace,
                                                   error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(evaluation.plan, cache_envelope.descriptor,
                                                     positions.data(), kGeometryGeneration,
                                                     charges.data(), cache_alias, local_workspace,
                                                     error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(same_cache_descriptor(cache_envelope.descriptor, descriptor_before));
    CHECK(tail_is(cache_envelope, kTailSentinel));

    local_workspace = evaluation.workspace;
    local_workspace.matrix_scratch = cache_alias;
    const ES2Workspace matrix_workspace_before = local_workspace;
    CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
              evaluation.plan, positions.data(), kGeometryGeneration + 1u, matrix_output.data(),
              matrix_output.size(), local_workspace, cache_envelope.descriptor,
              error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(same_workspace_descriptor(local_workspace, matrix_workspace_before));

    local_workspace = evaluation.workspace;
    local_workspace.shell_scratch = cache_alias;
    const ES2Workspace shell_workspace_before = local_workspace;
    CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
              evaluation.plan, cache_envelope.descriptor, charges.data(), potential.data(),
              local_workspace, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(same_workspace_descriptor(local_workspace, shell_workspace_before));

    local_workspace = evaluation.workspace;
    local_workspace.batch_scratch = cache_alias;
    const ES2Workspace batch_workspace_before = local_workspace;
    CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(evaluation.plan, cache_envelope.descriptor,
                                                   charges.data(), energy.data(), local_workspace,
                                                   error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(same_workspace_descriptor(local_workspace, batch_workspace_before));

    local_workspace = evaluation.workspace;
    local_workspace.gradient_scratch = cache_alias;
    const ES2Workspace gradient_workspace_before = local_workspace;
    CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(
              evaluation.plan, cache_envelope.descriptor, positions.data(), kGeometryGeneration,
              charges.data(), gradient.data(), local_workspace,
              error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(same_workspace_descriptor(local_workspace, gradient_workspace_before));
    CHECK(same_cache_descriptor(cache_envelope.descriptor, descriptor_before));
    CHECK(tail_is(cache_envelope, kTailSentinel));
    CHECK(outputs_are_unchanged(matrix_output, potential, energy, gradient));

    /* A cache backing pointer into its own descriptor is invalid for every API. */
    cache_envelope.descriptor = evaluation.cache;
    cache_envelope.descriptor.coulomb_matrix = cache_alias;
    descriptor_before = cache_envelope.descriptor;
    local_workspace = evaluation.workspace;
    CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
              evaluation.plan, positions.data(), kGeometryGeneration + 1u, matrix_output.data(),
              matrix_output.size(), local_workspace, cache_envelope.descriptor,
              error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
              evaluation.plan, cache_envelope.descriptor, charges.data(), potential.data(),
              local_workspace, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(evaluation.plan, cache_envelope.descriptor,
                                                   charges.data(), energy.data(), local_workspace,
                                                   error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(
              evaluation.plan, cache_envelope.descriptor, positions.data(), kGeometryGeneration,
              charges.data(), gradient.data(), local_workspace,
              error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(same_cache_descriptor(cache_envelope.descriptor, descriptor_before));
    CHECK(tail_is(cache_envelope, kTailSentinel));
    CHECK(outputs_are_unchanged(matrix_output, potential, energy, gradient));

    /* Repeat the output, scratch, and forged-cache matrix for workspace storage. */
    DescriptorEnvelope<ES2Workspace> workspace_envelope;
    workspace_envelope.descriptor = evaluation.workspace;
    workspace_envelope.tail.fill(kTailSentinel);
    double* workspace_alias = alias_at(&workspace_envelope.descriptor, alias_offset);
    ES2GeometryCache local_cache = evaluation.cache;
    const ES2GeometryCache local_cache_before = local_cache;
    ES2Workspace descriptor_workspace_before = workspace_envelope.descriptor;

    CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
              evaluation.plan, positions.data(), kGeometryGeneration + 1u, workspace_alias,
              evaluation.matrix.size(), workspace_envelope.descriptor, local_cache,
              error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
              evaluation.plan, local_cache, charges.data(), workspace_alias,
              workspace_envelope.descriptor, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(evaluation.plan, local_cache, charges.data(),
                                                   workspace_alias, workspace_envelope.descriptor,
                                                   error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(evaluation.plan, local_cache, positions.data(),
                                                     kGeometryGeneration, charges.data(),
                                                     workspace_alias, workspace_envelope.descriptor,
                                                     error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(same_cache_descriptor(local_cache, local_cache_before));
    CHECK(same_workspace_descriptor(workspace_envelope.descriptor, descriptor_workspace_before));
    CHECK(tail_is(workspace_envelope, kTailSentinel));

    workspace_envelope.descriptor = evaluation.workspace;
    workspace_envelope.descriptor.matrix_scratch = workspace_alias;
    descriptor_workspace_before = workspace_envelope.descriptor;
    CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
              evaluation.plan, positions.data(), kGeometryGeneration + 1u, matrix_output.data(),
              matrix_output.size(), workspace_envelope.descriptor, local_cache,
              error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(same_workspace_descriptor(workspace_envelope.descriptor, descriptor_workspace_before));

    workspace_envelope.descriptor = evaluation.workspace;
    workspace_envelope.descriptor.shell_scratch = workspace_alias;
    descriptor_workspace_before = workspace_envelope.descriptor;
    CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
              evaluation.plan, local_cache, charges.data(), potential.data(),
              workspace_envelope.descriptor, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(same_workspace_descriptor(workspace_envelope.descriptor, descriptor_workspace_before));

    workspace_envelope.descriptor = evaluation.workspace;
    workspace_envelope.descriptor.batch_scratch = workspace_alias;
    descriptor_workspace_before = workspace_envelope.descriptor;
    CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(evaluation.plan, local_cache, charges.data(),
                                                   energy.data(), workspace_envelope.descriptor,
                                                   error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(same_workspace_descriptor(workspace_envelope.descriptor, descriptor_workspace_before));

    workspace_envelope.descriptor = evaluation.workspace;
    workspace_envelope.descriptor.gradient_scratch = workspace_alias;
    descriptor_workspace_before = workspace_envelope.descriptor;
    CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(evaluation.plan, local_cache, positions.data(),
                                                     kGeometryGeneration, charges.data(),
                                                     gradient.data(), workspace_envelope.descriptor,
                                                     error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(same_workspace_descriptor(workspace_envelope.descriptor, descriptor_workspace_before));
    CHECK(tail_is(workspace_envelope, kTailSentinel));
    CHECK(outputs_are_unchanged(matrix_output, potential, energy, gradient));

    workspace_envelope.descriptor = evaluation.workspace;
    ES2GeometryCache forged_cache = evaluation.cache;
    forged_cache.coulomb_matrix = workspace_alias;
    const ES2GeometryCache forged_before = forged_cache;
    descriptor_workspace_before = workspace_envelope.descriptor;
    CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
              evaluation.plan, positions.data(), kGeometryGeneration + 1u, matrix_output.data(),
              matrix_output.size(), workspace_envelope.descriptor, forged_cache,
              error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
              evaluation.plan, forged_cache, charges.data(), potential.data(),
              workspace_envelope.descriptor, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(evaluation.plan, forged_cache, charges.data(),
                                                   energy.data(), workspace_envelope.descriptor,
                                                   error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(
              evaluation.plan, forged_cache, positions.data(), kGeometryGeneration, charges.data(),
              gradient.data(), workspace_envelope.descriptor,
              error) == GPUXTB_STATUS_INVALID_ARGUMENT);
    CHECK(same_cache_descriptor(forged_cache, forged_before));
    CHECK(same_workspace_descriptor(workspace_envelope.descriptor, descriptor_workspace_before));
    CHECK(tail_is(workspace_envelope, kTailSentinel));
    CHECK(outputs_are_unchanged(matrix_output, potential, energy, gradient));
  }

  CHECK(same_cache_descriptor(evaluation.cache, cache_before));
  CHECK(same_workspace_descriptor(evaluation.workspace, workspace_before));
  CHECK(evaluation.plan.identity() == identity_before);
  CHECK(evaluation.plan.atom_offsets() == atom_offsets_before);
  CHECK(evaluation.plan.shell_hardness() == hardness_before);
  return 0;
}

int test_validation_and_strong_guarantees() {
  const std::vector<std::int64_t> offsets{0, 2};
  const std::vector<std::int32_t> atomic_numbers{1, 1};
  const std::vector<double> valid_positions{0.0, 0.0, 0.0, 1.4, 0.0, 0.0};
  Evaluation evaluation;
  std::string error;
  CHECK(make_evaluation(offsets, atomic_numbers, valid_positions, evaluation, error));

  const std::vector<std::int32_t> wrong_atomic_numbers{1, 8};
  ES2Plan untouched_plan = evaluation.plan;
  CHECK(gpuxtb::detail::gfn2::make_es2_plan(evaluation.basis, wrong_atomic_numbers.data(),
                                            evaluation.plan,
                                            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(evaluation.plan.total_matrix_elements() == untouched_plan.total_matrix_elements());
  evaluation.plan = untouched_plan;

  std::vector<double> storage(evaluation.matrix.size(), 7.0);
  ES2GeometryCache old_cache = evaluation.cache;
  const double maximum = std::numeric_limits<double>::max();
  const std::vector<double> overflowing_positions{maximum, 0.0, 0.0, -maximum, 0.0, 0.0};
  CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
            evaluation.plan, overflowing_positions.data(), kGeometryGeneration + 1u, storage.data(),
            storage.size(), evaluation.workspace, evaluation.cache,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(storage.begin(), storage.end(), [](double value) { return value == 7.0; }));
  CHECK(evaluation.cache.coulomb_matrix == old_cache.coulomb_matrix);
  CHECK(evaluation.cache.matrix_elements == old_cache.matrix_elements);
  CHECK(evaluation.cache.geometry_generation == old_cache.geometry_generation);
  CHECK(evaluation.cache.plan_identity == old_cache.plan_identity);
  evaluation.cache = old_cache;
  CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
            evaluation.plan, valid_positions.data(), kGeometryGeneration + 1u, storage.data(),
            storage.size() - 1u, evaluation.workspace, evaluation.cache,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(storage.begin(), storage.end(), [](double value) { return value == 7.0; }));
  CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
            evaluation.plan, valid_positions.data(), kGeometryGeneration + 1u,
            const_cast<double*>(valid_positions.data()), evaluation.matrix.size(),
            evaluation.workspace, evaluation.cache, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  evaluation.cache = old_cache;

  const std::vector<double> charges{0.2, -0.3};
  std::array<double, 2> potential{9.0, 9.0};
  ES2GeometryCache corrupt_cache = evaluation.cache;
  const double original_off_diagonal = evaluation.matrix[1];
  evaluation.matrix[1] = std::numeric_limits<double>::quiet_NaN();
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
            evaluation.plan, corrupt_cache, charges.data(), potential.data(), evaluation.workspace,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(potential[0] == 9.0 && potential[1] == 9.0);
  evaluation.matrix[1] = original_off_diagonal;

  const std::vector<double> huge_charges{maximum, maximum};
  std::vector<double> overflowing_matrix(evaluation.matrix.size(), maximum);
  const std::vector<double> overflowing_charges{2.0, 2.0};
  ES2GeometryCache overflowing_cache{overflowing_matrix.data(),
                                     evaluation.plan.total_matrix_elements(), kGeometryGeneration,
                                     evaluation.plan.identity()};
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
            evaluation.plan, overflowing_cache, overflowing_charges.data(), potential.data(),
            evaluation.workspace, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(potential[0] == 9.0 && potential[1] == 9.0);
  std::array<double, 1> energy{4.0};
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(
            evaluation.plan, evaluation.cache, huge_charges.data(), energy.data(),
            evaluation.workspace, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(energy[0] == 4.0);
  std::array<double, 6> gradient{3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
  CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(
            evaluation.plan, evaluation.cache, valid_positions.data(), kGeometryGeneration,
            huge_charges.data(), gradient.data(), evaluation.workspace,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(gradient.begin(), gradient.end(), [](double value) { return value == 3.0; }));

  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(evaluation.plan, evaluation.cache, nullptr,
                                                         potential.data(), evaluation.workspace,
                                                         error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
            evaluation.plan, evaluation.cache, charges.data(), const_cast<double*>(charges.data()),
            evaluation.workspace, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  return 0;
}

int test_workspace_atomicity_overlap_and_generation() {
  const std::vector<std::int64_t> offsets{0, 2, 4};
  const std::vector<std::int32_t> atomic_numbers{1, 1, 1, 1};
  const std::vector<double> positions{
      0.0, 0.0, 0.0, 1.4, 0.0, 0.0, 3.0, 0.2, -0.1, 4.3, -0.4, 0.7,
  };
  Evaluation evaluation;
  std::string error;
  CHECK(make_evaluation(offsets, atomic_numbers, positions, evaluation, error));

  const std::vector<double> regular_charges{0.1, -0.2, 0.3, -0.4};
  std::vector<double> stale_gradient(positions.size(), 5.0);
  CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(
            evaluation.plan, evaluation.cache, positions.data(), kGeometryGeneration + 1u,
            regular_charges.data(), stale_gradient.data(), evaluation.workspace,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(stale_gradient.begin(), stale_gradient.end(),
                    [](double value) { return value == 5.0; }));

  const std::vector<double> matrix_before = evaluation.matrix;
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
            evaluation.plan, evaluation.cache, regular_charges.data(), evaluation.matrix.data() + 1,
            evaluation.workspace, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(evaluation.matrix == matrix_before);

  std::vector<double> partial_shell_arena(5, 6.0);
  ES2Workspace partial_shell_workspace = evaluation.workspace;
  partial_shell_workspace.shell_scratch = partial_shell_arena.data() + 1;
  partial_shell_workspace.shell_elements = evaluation.plan.total_shells();
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
            evaluation.plan, evaluation.cache, regular_charges.data(), partial_shell_arena.data(),
            partial_shell_workspace, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(partial_shell_arena.begin(), partial_shell_arena.end(),
                    [](double value) { return value == 6.0; }));

  std::vector<double> partial_matrix_arena(
      static_cast<std::size_t>(evaluation.plan.total_matrix_elements()) + 1u, 7.0);
  ES2Workspace partial_matrix_workspace = evaluation.workspace;
  partial_matrix_workspace.matrix_scratch = partial_matrix_arena.data() + 1;
  partial_matrix_workspace.matrix_elements = evaluation.plan.total_matrix_elements();
  ES2GeometryCache unpublished_cache;
  CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
            evaluation.plan, positions.data(), kGeometryGeneration + 1u,
            partial_matrix_arena.data(),
            static_cast<std::size_t>(evaluation.plan.total_matrix_elements()),
            partial_matrix_workspace, unpublished_cache, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(partial_matrix_arena.begin(), partial_matrix_arena.end(),
                    [](double value) { return value == 7.0; }));

  const double maximum = std::numeric_limits<double>::max();
  std::vector<double> later_overflow_positions = positions;
  later_overflow_positions[6] = maximum;
  later_overflow_positions[9] = -maximum;
  std::vector<double> unpublished_matrix(evaluation.matrix.size(), 8.0);
  const ES2GeometryCache cache_before = evaluation.cache;
  CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
            evaluation.plan, later_overflow_positions.data(), kGeometryGeneration + 1u,
            unpublished_matrix.data(), unpublished_matrix.size(), evaluation.workspace,
            evaluation.cache, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(unpublished_matrix.begin(), unpublished_matrix.end(),
                    [](double value) { return value == 8.0; }));
  CHECK(evaluation.cache.coulomb_matrix == cache_before.coulomb_matrix);
  CHECK(evaluation.cache.matrix_elements == cache_before.matrix_elements);
  CHECK(evaluation.cache.geometry_generation == cache_before.geometry_generation);
  CHECK(evaluation.cache.plan_identity == cache_before.plan_identity);

  std::vector<double> overflow_matrix = evaluation.matrix;
  const std::int64_t second_matrix_begin = evaluation.plan.matrix_offsets()[1];
  const std::int64_t second_matrix_end = evaluation.plan.matrix_offsets()[2];
  std::fill(overflow_matrix.begin() + second_matrix_begin,
            overflow_matrix.begin() + second_matrix_end, maximum);
  const ES2GeometryCache overflow_cache{overflow_matrix.data(),
                                        evaluation.plan.total_matrix_elements(),
                                        kGeometryGeneration, evaluation.plan.identity()};
  const std::vector<double> overflow_charges{0.1, -0.2, 2.0, 2.0};
  std::vector<double> potentials(4, 9.0);
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
            evaluation.plan, overflow_cache, overflow_charges.data(), potentials.data(),
            evaluation.workspace, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(
      std::all_of(potentials.begin(), potentials.end(), [](double value) { return value == 9.0; }));

  std::array<double, 2> energies{10.0, 10.0};
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(
            evaluation.plan, overflow_cache, overflow_charges.data(), energies.data(),
            evaluation.workspace, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(energies[0] == 10.0 && energies[1] == 10.0);

  const std::vector<double> gradient_overflow_charges{0.1, -0.2, maximum, maximum};
  std::vector<double> gradients(positions.size(), 11.0);
  CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(
            evaluation.plan, evaluation.cache, positions.data(), kGeometryGeneration,
            gradient_overflow_charges.data(), gradients.data(), evaluation.workspace,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(
      std::all_of(gradients.begin(), gradients.end(), [](double value) { return value == 11.0; }));
  return 0;
}

int test_misaligned_buffers_are_rejected_atomically() {
  const std::vector<std::int64_t> offsets{0, 2};
  const std::vector<std::int32_t> atomic_numbers{1, 1};
  const std::vector<double> positions{0.0, 0.0, 0.0, 1.4, 0.0, 0.0};
  const std::vector<double> charges{0.2, -0.3};
  Evaluation evaluation;
  std::string error;
  CHECK(make_evaluation(offsets, atomic_numbers, positions, evaluation, error));

  const std::size_t matrix_count =
      static_cast<std::size_t>(evaluation.plan.total_matrix_elements());
  const std::size_t shell_count = static_cast<std::size_t>(evaluation.plan.total_shells());
  const std::size_t batch_count = static_cast<std::size_t>(evaluation.plan.batch_size());
  const std::size_t gradient_count = static_cast<std::size_t>(evaluation.plan.total_atoms()) * 3u;
  MisalignedDoubles misaligned_matrix(matrix_count);
  MisalignedDoubles misaligned_shell(shell_count);
  MisalignedDoubles misaligned_batch(batch_count);
  MisalignedDoubles misaligned_gradient(gradient_count);
  CHECK(misaligned_matrix.is_misaligned());
  CHECK(misaligned_shell.is_misaligned());
  CHECK(misaligned_batch.is_misaligned());
  CHECK(misaligned_gradient.is_misaligned());

  std::vector<double> matrix_storage(matrix_count, 6.0);
  ES2GeometryCache cache_before = evaluation.cache;
  CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
            evaluation.plan, misaligned_gradient.data, kGeometryGeneration + 1u,
            matrix_storage.data(), matrix_storage.size(), evaluation.workspace, evaluation.cache,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(matrix_storage.begin(), matrix_storage.end(),
                    [](double value) { return value == 6.0; }));
  CHECK(evaluation.cache.coulomb_matrix == cache_before.coulomb_matrix);
  CHECK(evaluation.cache.matrix_elements == cache_before.matrix_elements);
  CHECK(evaluation.cache.geometry_generation == cache_before.geometry_generation);
  CHECK(evaluation.cache.plan_identity == cache_before.plan_identity);

  ES2GeometryCache unpublished_cache;
  CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
            evaluation.plan, positions.data(), kGeometryGeneration + 1u, misaligned_matrix.data,
            matrix_count, evaluation.workspace, unpublished_cache,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(misaligned_matrix.untouched());
  CHECK(unpublished_cache.coulomb_matrix == nullptr && unpublished_cache.matrix_elements == 0 &&
        unpublished_cache.geometry_generation == 0u && unpublished_cache.plan_identity == nullptr);

  std::array<double, 2> potential{9.0, 9.0};
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
            evaluation.plan, evaluation.cache, misaligned_shell.data, potential.data(),
            evaluation.workspace, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(potential[0] == 9.0 && potential[1] == 9.0);
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
            evaluation.plan, evaluation.cache, charges.data(), misaligned_shell.data,
            evaluation.workspace, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(misaligned_shell.untouched());

  std::array<double, 1> energy{8.0};
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(
            evaluation.plan, evaluation.cache, misaligned_shell.data, energy.data(),
            evaluation.workspace, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(energy[0] == 8.0);
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(evaluation.plan, evaluation.cache, charges.data(),
                                                 misaligned_batch.data, evaluation.workspace,
                                                 error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(misaligned_batch.untouched());

  std::array<double, 6> gradient{7.0, 7.0, 7.0, 7.0, 7.0, 7.0};
  CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(
            evaluation.plan, evaluation.cache, misaligned_gradient.data, kGeometryGeneration,
            charges.data(), gradient.data(), evaluation.workspace,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(gradient.begin(), gradient.end(), [](double value) { return value == 7.0; }));
  CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(
            evaluation.plan, evaluation.cache, positions.data(), kGeometryGeneration,
            misaligned_shell.data, gradient.data(), evaluation.workspace,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(gradient.begin(), gradient.end(), [](double value) { return value == 7.0; }));
  CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(
            evaluation.plan, evaluation.cache, positions.data(), kGeometryGeneration,
            charges.data(), misaligned_gradient.data, evaluation.workspace,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(misaligned_gradient.untouched());

  const ES2GeometryCache misaligned_cache{misaligned_matrix.data,
                                          evaluation.plan.total_matrix_elements(),
                                          kGeometryGeneration, evaluation.plan.identity()};
  potential = {9.0, 9.0};
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
            evaluation.plan, misaligned_cache, charges.data(), potential.data(),
            evaluation.workspace, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(potential[0] == 9.0 && potential[1] == 9.0);

  ES2Workspace bad_workspace = evaluation.workspace;
  bad_workspace.matrix_scratch = misaligned_matrix.data;
  CHECK(gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
            evaluation.plan, positions.data(), kGeometryGeneration + 1u, matrix_storage.data(),
            matrix_storage.size(), bad_workspace, unpublished_cache,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(matrix_storage.begin(), matrix_storage.end(),
                    [](double value) { return value == 6.0; }));
  bad_workspace = evaluation.workspace;
  bad_workspace.shell_scratch = misaligned_shell.data;
  potential = {9.0, 9.0};
  CHECK(gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
            evaluation.plan, evaluation.cache, charges.data(), potential.data(), bad_workspace,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(potential[0] == 9.0 && potential[1] == 9.0);
  bad_workspace = evaluation.workspace;
  bad_workspace.batch_scratch = misaligned_batch.data;
  energy[0] = 8.0;
  CHECK(gpuxtb::detail::gfn2::add_es2_energy_cpu(evaluation.plan, evaluation.cache, charges.data(),
                                                 energy.data(), bad_workspace,
                                                 error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(energy[0] == 8.0);
  bad_workspace = evaluation.workspace;
  bad_workspace.gradient_scratch = misaligned_gradient.data;
  gradient.fill(7.0);
  CHECK(gpuxtb::detail::gfn2::add_es2_gradient_cpu(evaluation.plan, evaluation.cache,
                                                   positions.data(), kGeometryGeneration,
                                                   charges.data(), gradient.data(), bad_workspace,
                                                   error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(gradient.begin(), gradient.end(), [](double value) { return value == 7.0; }));
  return 0;
}

int test_no_steady_state_allocations() {
  const std::vector<std::int64_t> offsets{0, 3};
  const std::vector<std::int32_t> atomic_numbers{6, 8, 1};
  const std::vector<double> positions{0.1, -0.2, 0.4, 1.5, 0.3, -0.7, -0.8, 1.2, 0.9};
  Evaluation evaluation;
  std::string error;
  CHECK(make_evaluation(offsets, atomic_numbers, positions, evaluation, error));
  std::vector<double> charges(static_cast<std::size_t>(evaluation.plan.total_shells()), 0.12);
  std::vector<double> potential(charges.size());
  std::array<double, 1> energy{};
  std::vector<double> gradient(positions.size());
  error.reserve(128);

  const std::size_t before = allocation_test::count.load(std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const gpuxtb_status_t cache_status = gpuxtb::detail::gfn2::update_es2_geometry_cache_cpu(
      evaluation.plan, positions.data(), kGeometryGeneration + 1u, evaluation.matrix.data(),
      evaluation.matrix.size(), evaluation.workspace, evaluation.cache, error);
  const gpuxtb_status_t potential_status = gpuxtb::detail::gfn2::evaluate_es2_potential_cpu(
      evaluation.plan, evaluation.cache, charges.data(), potential.data(), evaluation.workspace,
      error);
  const gpuxtb_status_t energy_status =
      gpuxtb::detail::gfn2::add_es2_energy_cpu(evaluation.plan, evaluation.cache, charges.data(),
                                               energy.data(), evaluation.workspace, error);
  double system_energy = 0.0;
  const gpuxtb_status_t system_energy_status = gpuxtb::detail::gfn2::add_es2_energy_system_cpu(
      evaluation.plan, evaluation.cache, 0, charges.data(), system_energy, error);
  const gpuxtb_status_t gradient_status = gpuxtb::detail::gfn2::add_es2_gradient_cpu(
      evaluation.plan, evaluation.cache, positions.data(), kGeometryGeneration + 1u, charges.data(),
      gradient.data(), evaluation.workspace, error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  const std::size_t after = allocation_test::count.load(std::memory_order_relaxed);

  CHECK(cache_status == GPUXTB_STATUS_SUCCESS);
  CHECK(potential_status == GPUXTB_STATUS_SUCCESS);
  CHECK(energy_status == GPUXTB_STATUS_SUCCESS);
  CHECK(system_energy_status == GPUXTB_STATUS_SUCCESS);
  CHECK(gradient_status == GPUXTB_STATUS_SUCCESS);
  CHECK(after == before);
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_tblite_component_oracle(); status != 0) {
    return status;
  }
  if (const int status = test_long_range_and_heavy_element(); status != 0) {
    return status;
  }
  if (const int status = test_energy_potential_charge_derivative(); status != 0) {
    return status;
  }
  if (const int status = test_coordinate_vjp(); status != 0) {
    return status;
  }
  if (const int status = test_ragged_matches_sequential(); status != 0) {
    return status;
  }
  if (const int status = test_system_energy_failure_isolation_and_binding(); status != 0) {
    return status;
  }
  if (const int status = test_system_potential_matches_batch_and_failure_isolation(); status != 0) {
    return status;
  }
  if (const int status = test_cache_plan_identity(); status != 0) {
    return status;
  }
  if (const int status = test_default_and_moved_from_plans_are_rejected_atomically(); status != 0) {
    return status;
  }
  if (const int status = test_plan_storage_aliases_are_rejected_atomically(); status != 0) {
    return status;
  }
  if (const int status = test_opaque_plan_object_aliases_and_active_cache_descriptor();
      status != 0) {
    return status;
  }
  if (const int status = test_descriptor_object_alias_matrix(); status != 0) {
    return status;
  }
  if (const int status = test_validation_and_strong_guarantees(); status != 0) {
    return status;
  }
  if (const int status = test_workspace_atomicity_overlap_and_generation(); status != 0) {
    return status;
  }
  if (const int status = test_misaligned_buffers_are_rejected_atomically(); status != 0) {
    return status;
  }
  return test_no_steady_state_allocations();
}
