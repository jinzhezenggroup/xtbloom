// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "data/parameters/gfn1_d3.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "data/parameters/gfn1.hpp"
#include "model/gfn1/coordination.hpp"
#include "model/gfn1/d3.hpp"

namespace {

using xtbloom::detail::gfn1::CoordinationPlan;
using xtbloom::detail::gfn1::D3Plan;
using xtbloom::detail::gfn1::D3Workspace;

int failures = 0;

void expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void expect_near(double actual, double expected, double tolerance, const char* message) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " actual=" << actual << " expected=" << expected
              << " tolerance=" << tolerance << '\n';
    ++failures;
  }
}

class AlignedBuffer {
 public:
  explicit AlignedBuffer(std::size_t size_bytes)
      : size_bytes_(std::max(size_bytes, xtbloom::detail::gfn1::kD3WorkspaceAlignment)),
        data_(::operator new(size_bytes_,
                             std::align_val_t(xtbloom::detail::gfn1::kD3WorkspaceAlignment))) {}

  ~AlignedBuffer() {
    ::operator delete(data_, std::align_val_t(xtbloom::detail::gfn1::kD3WorkspaceAlignment));
  }

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  [[nodiscard]] void* data() const { return data_; }
  [[nodiscard]] std::size_t size_bytes() const { return size_bytes_; }

 private:
  std::size_t size_bytes_;
  void* data_;
};

struct Fixture {
  std::vector<std::int64_t> offsets;
  std::vector<std::int32_t> numbers;
  std::vector<double> positions;
  D3Plan d3_plan;
  CoordinationPlan coordination_plan;
  AlignedBuffer storage;
  D3Workspace workspace;

  Fixture(std::vector<std::int64_t> atom_offsets, std::vector<std::int32_t> atomic_numbers,
          std::vector<double> coordinates)
      : offsets(std::move(atom_offsets)),
        numbers(std::move(atomic_numbers)),
        positions(std::move(coordinates)),
        storage(temporary_workspace_size(offsets, numbers)) {
    std::string error;
    const auto batch_size = static_cast<std::int64_t>(offsets.size() - 1u);
    const auto total_atoms = static_cast<std::int64_t>(numbers.size());
    expect(
        xtbloom::detail::gfn1::make_d3_plan(batch_size, total_atoms, offsets.data(), numbers.data(),
                                            d3_plan, error) == XTBLOOM_STATUS_SUCCESS,
        "make D3 fixture plan");
    expect(xtbloom::detail::gfn1::make_coordination_plan(batch_size, total_atoms, offsets.data(),
                                                         numbers.data(), coordination_plan,
                                                         error) == XTBLOOM_STATUS_SUCCESS,
           "make D3 coordination fixture plan");
    expect(xtbloom::detail::gfn1::bind_d3_workspace(d3_plan, storage.data(), storage.size_bytes(),
                                                    workspace, error) == XTBLOOM_STATUS_SUCCESS,
           "bind D3 fixture workspace");
  }

  static std::size_t temporary_workspace_size(const std::vector<std::int64_t>& offsets,
                                              const std::vector<std::int32_t>& numbers) {
    D3Plan plan;
    std::string error;
    const auto status = xtbloom::detail::gfn1::make_d3_plan(
        static_cast<std::int64_t>(offsets.size() - 1u), static_cast<std::int64_t>(numbers.size()),
        offsets.data(), numbers.data(), plan, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      std::cerr << "FAIL: preliminary D3 plan: " << error << '\n';
      ++failures;
      return xtbloom::detail::gfn1::kD3WorkspaceAlignment;
    }
    return plan.workspace_size_bytes();
  }

  [[nodiscard]] std::vector<double> coordination(const std::vector<double>& geometry) const {
    std::vector<double> values(numbers.size(), -1.0);
    std::string error;
    expect(xtbloom::detail::gfn1::evaluate_coordination_cpu(
               coordination_plan, geometry.data(), values.data(), error) == XTBLOOM_STATUS_SUCCESS,
           "evaluate fixture coordination");
    return values;
  }

  [[nodiscard]] double energy(const std::vector<double>& geometry,
                              const std::vector<double>* explicit_cn = nullptr) {
    const std::vector<double> computed =
        explicit_cn == nullptr ? coordination(geometry) : std::vector<double>();
    const double* cn = explicit_cn == nullptr ? computed.data() : explicit_cn->data();
    std::vector<double> energies(offsets.size() - 1u, 0.0);
    std::string error;
    expect(xtbloom::detail::gfn1::add_d3_cpu(d3_plan, geometry.data(), cn, energies.data(), nullptr,
                                             workspace, error) == XTBLOOM_STATUS_SUCCESS,
           "evaluate D3 fixture energy");
    double total = 0.0;
    for (double value : energies) {
      total += value;
    }
    return total;
  }

  [[nodiscard]] std::vector<double> gradient(const std::vector<double>& geometry) {
    const std::vector<double> cn = coordination(geometry);
    std::vector<double> energies(offsets.size() - 1u, 0.0);
    std::vector<double> values(geometry.size(), 0.0);
    std::string error;
    expect(xtbloom::detail::gfn1::add_d3_cpu(d3_plan, geometry.data(), cn.data(), energies.data(),
                                             values.data(), workspace,
                                             error) == XTBLOOM_STATUS_SUCCESS,
           "evaluate D3 fixture gradient");
    return values;
  }
};

struct ReferenceWeights {
  std::array<double, xtbloom::detail::gfn1::kD3MaximumReferences> value{};
  std::array<double, xtbloom::detail::gfn1::kD3MaximumReferences> derivative{};
};

ReferenceWeights independent_weights(std::uint32_t atomic_number, double coordination) {
  ReferenceWeights result;
  const auto& element = xtbloom::parameters::gfn1_d3::kElements[atomic_number - 1u];
  double norm = 0.0;
  double derivative_norm = 0.0;
  double maximum_cn = -std::numeric_limits<double>::infinity();
  for (std::size_t reference = 0; reference < element.reference_count; ++reference) {
    const double cn = xtbloom::parameters::gfn1_d3::reference_cn(
        atomic_number, static_cast<std::uint32_t>(reference));
    maximum_cn = std::max(maximum_cn, cn);
    const double delta = cn - coordination;
    result.value[reference] = std::exp(-4.0 * delta * delta);
    norm += result.value[reference];
    derivative_norm += 8.0 * delta * result.value[reference];
  }
  const double inverse_norm = 1.0 / norm;
  for (std::size_t reference = 0; reference < element.reference_count; ++reference) {
    const double cn = xtbloom::parameters::gfn1_d3::reference_cn(
        atomic_number, static_cast<std::uint32_t>(reference));
    const double unnormalized = result.value[reference];
    const double delta = cn - coordination;
    result.value[reference] = unnormalized * inverse_norm;
    if (!std::isfinite(result.value[reference])) {
      result.value[reference] = cn == maximum_cn ? 1.0 : 0.0;
    }
    result.derivative[reference] = 8.0 * delta * unnormalized * inverse_norm -
                                   unnormalized * derivative_norm * inverse_norm * inverse_norm;
    if (!std::isfinite(result.derivative[reference])) {
      result.derivative[reference] = 0.0;
    }
  }
  return result;
}

struct IndependentPair {
  double c6 = 0.0;
  double first_derivative = 0.0;
  double second_derivative = 0.0;
};

IndependentPair independent_c6(std::uint32_t first, double first_cn, std::uint32_t second,
                               double second_cn) {
  const ReferenceWeights first_weights = independent_weights(first, first_cn);
  const ReferenceWeights second_weights = independent_weights(second, second_cn);
  const auto& first_element = xtbloom::parameters::gfn1_d3::kElements[first - 1u];
  const auto& second_element = xtbloom::parameters::gfn1_d3::kElements[second - 1u];
  IndependentPair result;
  for (std::size_t first_reference = 0; first_reference < first_element.reference_count;
       ++first_reference) {
    for (std::size_t second_reference = 0; second_reference < second_element.reference_count;
         ++second_reference) {
      const double c6 = xtbloom::parameters::gfn1_d3::reference_c6(
          first, static_cast<std::uint32_t>(first_reference), second,
          static_cast<std::uint32_t>(second_reference));
      result.c6 +=
          first_weights.value[first_reference] * second_weights.value[second_reference] * c6;
      result.first_derivative +=
          first_weights.derivative[first_reference] * second_weights.value[second_reference] * c6;
      result.second_derivative +=
          first_weights.value[first_reference] * second_weights.derivative[second_reference] * c6;
    }
  }
  return result;
}

double independent_pair_energy(std::uint32_t first, double first_cn, std::uint32_t second,
                               double second_cn, double distance) {
  const IndependentPair coefficient = independent_c6(first, first_cn, second, second_cn);
  const double q = 3.0 * xtbloom::parameters::gfn1_d3::kR4R2[first - 1u] *
                   xtbloom::parameters::gfn1_d3::kR4R2[second - 1u];
  const double r0 = xtbloom::parameters::gfn1::kGlobal.dispersion_a1 * std::sqrt(q) +
                    xtbloom::parameters::gfn1::kGlobal.dispersion_a2;
  const double r2 = distance * distance;
  const double r4 = r2 * r2;
  const double r0_2 = r0 * r0;
  const double r0_4 = r0_2 * r0_2;
  const double r0_6 = r0_4 * r0_2;
  const double r0_8 = r0_4 * r0_4;
  double sw = 1.0;
  if (distance >= 50.0) {
    sw = 0.0;
  } else if (distance > 49.95) {
    const double x = (50.0 - distance) / 0.05;
    sw = x * x * x * (10.0 - 15.0 * x + 6.0 * x * x);
  }
  const double damping = sw * (1.0 / (r4 * r2 + r0_6) + 2.4 * q / (r4 * r4 + r0_8));
  return -coefficient.c6 * damping;
}

void test_plan_workspace_and_ragged_energy() {
  Fixture fixture({0, 2, 2, 5}, {1, 6, 8, 1, 86},
                  {0.0, 0.0, 0.0, 2.4, 0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.1, 0.4, -0.1});
  expect(fixture.d3_plan.sealed(), "D3 plan is sealed");
  expect(fixture.d3_plan.batch_size() == 3, "D3 plan batch size");
  expect(fixture.d3_plan.total_atoms() == 5, "D3 plan atom count");
  expect(fixture.d3_plan.total_pairs() == 4, "D3 plan ragged pair count");
  expect(fixture.d3_plan.pair_offsets() == std::vector<std::int64_t>({0, 1, 1, 4}),
         "D3 plan pair offsets");
  expect(fixture.d3_plan.matches_atomic_numbers(fixture.numbers.data()),
         "D3 plan matches atomic numbers");
  auto changed = fixture.numbers;
  changed.back() = 85;
  expect(!fixture.d3_plan.matches_atomic_numbers(changed.data()),
         "D3 plan rejects changed atomic numbers");
  expect(
      fixture.d3_plan.workspace_size_bytes() % xtbloom::detail::gfn1::kD3WorkspaceAlignment == 0u,
      "D3 workspace size is aligned");

  const auto cn = fixture.coordination(fixture.positions);
  std::array<double, 3> energies{{0.25, -0.5, 0.75}};
  std::string error;
  expect(xtbloom::detail::gfn1::add_d3_cpu(fixture.d3_plan, fixture.positions.data(), cn.data(),
                                           energies.data(), nullptr, fixture.workspace,
                                           error) == XTBLOOM_STATUS_SUCCESS,
         "ragged D3 energy succeeds");
  expect(energies[0] < 0.25, "first ragged D3 member contributes");
  expect(energies[1] == -0.5, "empty ragged D3 member is unchanged");
  expect(energies[2] < 0.75, "last ragged D3 member contributes");
}

void test_reference_interpolation_layout_and_fallback() {
  const std::vector<double> explicit_cn{1.2, 2.7};
  Fixture asymmetric({0, 2}, {1, 6}, {0.0, 0.0, 0.0, 4.25, 0.0, 0.0});
  const double expected = independent_pair_energy(1, explicit_cn[0], 6, explicit_cn[1], 4.25);
  expect_near(asymmetric.energy(asymmetric.positions, &explicit_cn), expected, 2.0e-15,
              "asymmetric reference layout matches independent interpolation");

  const std::vector<double> swapped_cn{2.7, 1.2};
  Fixture swapped({0, 2}, {6, 1}, {0.0, 0.0, 0.0, 4.25, 0.0, 0.0});
  expect_near(swapped.energy(swapped.positions, &swapped_cn), expected, 2.0e-15,
              "element order does not transpose packed C6 references");

  const std::vector<double> underflow_cn{1.0e200, 1.0e200};
  Fixture fallback({0, 2}, {6, 8}, {0.0, 0.0, 0.0, 5.5, 0.0, 0.0});
  const double fallback_expected =
      independent_pair_energy(6, underflow_cn[0], 8, underflow_cn[1], 5.5);
  expect_near(fallback.energy(fallback.positions, &underflow_cn), fallback_expected, 2.0e-15,
              "underflow fallback selects maximum reference CN");
}

// XTBLOOM_GFN1_FIXTURE_BEGIN gfn1-d3-dxtb-tblite
void test_dxtb_tblite_term_goldens() {
  /*
   * dxtb b529b5dd retains these GFN1 D3 values as tblite-generated
   * references. The coordinates come from its separately pinned LiH and
   * SiH4 coord files; both source records are authenticated by the repository
   * fixture manifest.
   */
  Fixture lih({0, 2}, {3, 1}, {0.0, 0.0, -1.50796743897235, 0.0, 0.0, 1.50796743897235});
  expect_near(lih.energy(lih.positions), -8.2108039012179698e-5, 2.0e-15,
              "LiH dxtb/tblite D3 energy golden");
  const std::vector<double> lih_gradient = lih.gradient(lih.positions);
  const std::array<double, 6> lih_reference{
      {0.0, 0.0, 2.35781197246301e-6, 0.0, 0.0, -2.35781197246301e-6}};
  for (std::size_t coordinate = 0; coordinate < lih_reference.size(); ++coordinate) {
    expect_near(lih_gradient[coordinate], lih_reference[coordinate], 2.0e-14,
                "LiH dxtb/tblite D3 gradient golden");
  }

  Fixture sih4(
      {0, 5}, {14, 1, 1, 1, 1},
      {0.0, 0.0, 0.0, 1.61768389755830, 1.61768389755830, -1.61768389755830, -1.61768389755830,
       -1.61768389755830, -1.61768389755830, 1.61768389755830, -1.61768389755830, 1.61768389755830,
       -1.61768389755830, 1.61768389755830, 1.61768389755830});
  expect_near(sih4.energy(sih4.positions), -6.8049872510979868e-4, 2.0e-15,
              "SiH4 dxtb/tblite D3 energy golden");
  const std::vector<double> sih4_gradient = sih4.gradient(sih4.positions);
  const std::array<double, 15> sih4_reference{
      {0.0, 0.0, 0.0, -7.37831467548165e-7, -7.37831467548165e-7, 7.37831467548165e-7,
       7.37831467548165e-7, 7.37831467548165e-7, 7.37831467548165e-7, -7.37831467548165e-7,
       7.37831467548165e-7, -7.37831467548165e-7, 7.37831467548165e-7, -7.37831467548165e-7,
       -7.37831467548165e-7}};
  for (std::size_t coordinate = 0; coordinate < sih4_reference.size(); ++coordinate) {
    expect_near(sih4_gradient[coordinate], sih4_reference[coordinate], 2.0e-14,
                "SiH4 dxtb/tblite D3 gradient golden");
  }
}
// XTBLOOM_GFN1_FIXTURE_END gfn1-d3-dxtb-tblite

void test_cutoff_contract() {
  const std::vector<double> cn{0.0, 0.0};
  for (double distance : {49.949999, 49.95, 49.975, 49.999999, 50.0, 50.000001}) {
    Fixture fixture({0, 2}, {6, 8}, {0.0, 0.0, 0.0, distance, 0.0, 0.0});
    const double expected = independent_pair_energy(6, 0.0, 8, 0.0, distance);
    expect_near(fixture.energy(fixture.positions, &cn), expected, 2.0e-15,
                "D3 smooth cutoff value");
  }
}

void test_gradients_and_invariance() {
  Fixture fixture({0, 4}, {6, 1, 8, 53},
                  {0.2, -0.3, 0.4, 2.1, 0.5, -0.7, -1.3, 2.4, 0.8, 3.2, -1.7, 1.1});
  const std::vector<double> analytic = fixture.gradient(fixture.positions);
  for (double step : {2.0e-4, 7.0e-5, 2.0e-5}) {
    double maximum_error = 0.0;
    for (std::size_t coordinate = 0; coordinate < fixture.positions.size(); ++coordinate) {
      auto plus = fixture.positions;
      auto minus = fixture.positions;
      plus[coordinate] += step;
      minus[coordinate] -= step;
      const double numerical = (fixture.energy(plus) - fixture.energy(minus)) / (2.0 * step);
      maximum_error = std::max(maximum_error, std::abs(numerical - analytic[coordinate]));
    }
    expect(maximum_error < 2.0e-9, "D3 complete gradient matches central differences");
  }

  std::array<double, 3> net_gradient{};
  std::array<double, 3> torque{};
  for (std::size_t atom = 0; atom < fixture.numbers.size(); ++atom) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      net_gradient[axis] += analytic[atom * 3u + axis];
    }
    const double x = fixture.positions[atom * 3u];
    const double y = fixture.positions[atom * 3u + 1u];
    const double z = fixture.positions[atom * 3u + 2u];
    const double gx = analytic[atom * 3u];
    const double gy = analytic[atom * 3u + 1u];
    const double gz = analytic[atom * 3u + 2u];
    torque[0] += y * gz - z * gy;
    torque[1] += z * gx - x * gz;
    torque[2] += x * gy - y * gx;
  }
  for (std::size_t axis = 0; axis < 3; ++axis) {
    expect_near(net_gradient[axis], 0.0, 2.0e-14, "D3 net gradient conservation");
    expect_near(torque[axis], 0.0, 5.0e-14, "D3 torque conservation");
  }

  auto translated = fixture.positions;
  for (std::size_t atom = 0; atom < fixture.numbers.size(); ++atom) {
    translated[atom * 3u] += 7.25;
    translated[atom * 3u + 1u] -= 3.5;
    translated[atom * 3u + 2u] += 1.75;
  }
  expect_near(fixture.energy(translated), fixture.energy(fixture.positions), 2.0e-15,
              "D3 energy is translation invariant");

  auto rotated = fixture.positions;
  const double angle = 0.37;
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  for (std::size_t atom = 0; atom < fixture.numbers.size(); ++atom) {
    const double x = fixture.positions[atom * 3u];
    const double y = fixture.positions[atom * 3u + 1u];
    rotated[atom * 3u] = cosine * x - sine * y;
    rotated[atom * 3u + 1u] = sine * x + cosine * y;
  }
  expect_near(fixture.energy(rotated), fixture.energy(fixture.positions), 2.0e-15,
              "D3 energy is rotation invariant");
}

void test_switch_gradient() {
  Fixture fixture({0, 2}, {6, 8}, {0.0, 0.0, 0.0, 49.975, 0.0, 0.0});
  const std::vector<double> analytic = fixture.gradient(fixture.positions);
  for (double step : {2.0e-5, 7.0e-6, 2.0e-6}) {
    auto plus = fixture.positions;
    auto minus = fixture.positions;
    plus[3] += step;
    minus[3] -= step;
    const double numerical = (fixture.energy(plus) - fixture.energy(minus)) / (2.0 * step);
    expect_near(analytic[3], numerical, 2.0e-10, "D3 switch gradient finite difference");
  }
}

void test_transactional_failures() {
  Fixture fixture({0, 2}, {6, 8}, {0.0, 0.0, 0.0, 4.0, 0.0, 0.0});
  const std::vector<double> cn = fixture.coordination(fixture.positions);
  std::array<double, 1> energies{{3.5}};
  std::array<double, 6> gradients{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
  const auto original_energies = energies;
  const auto original_gradients = gradients;
  std::string error;

  auto bad_positions = fixture.positions;
  bad_positions[0] = std::numeric_limits<double>::quiet_NaN();
  expect(xtbloom::detail::gfn1::add_d3_cpu(fixture.d3_plan, bad_positions.data(), cn.data(),
                                           energies.data(), gradients.data(), fixture.workspace,
                                           error) == XTBLOOM_STATUS_INVALID_ARGUMENT,
         "D3 rejects non-finite positions");
  expect(energies == original_energies && gradients == original_gradients,
         "D3 non-finite failure is transactional");

  std::vector<double> bad_cn = cn;
  bad_cn[1] = std::numeric_limits<double>::infinity();
  expect(xtbloom::detail::gfn1::add_d3_cpu(fixture.d3_plan, fixture.positions.data(), bad_cn.data(),
                                           energies.data(), gradients.data(), fixture.workspace,
                                           error) == XTBLOOM_STATUS_INVALID_ARGUMENT,
         "D3 rejects non-finite coordination");
  expect(energies == original_energies && gradients == original_gradients,
         "D3 coordination failure is transactional");

  D3Workspace tampered = fixture.workspace;
  tampered.weights += 1;
  expect(xtbloom::detail::gfn1::add_d3_cpu(fixture.d3_plan, fixture.positions.data(), cn.data(),
                                           energies.data(), gradients.data(), tampered,
                                           error) == XTBLOOM_STATUS_INVALID_ARGUMENT,
         "D3 rejects tampered workspace pointers");
  expect(energies == original_energies && gradients == original_gradients,
         "D3 workspace failure is transactional");

  std::array<double, 6> overlapping_outputs{{3.5, 1.0, 2.0, 3.0, 4.0, 5.0}};
  expect(xtbloom::detail::gfn1::add_d3_cpu(fixture.d3_plan, fixture.positions.data(), cn.data(),
                                           overlapping_outputs.data(), overlapping_outputs.data(),
                                           fixture.workspace,
                                           error) == XTBLOOM_STATUS_INVALID_ARGUMENT,
         "D3 rejects overlapping output accumulators");
  expect(energies == original_energies, "D3 alias failure preserves energy");
}

void test_invalid_plans_and_workspace_binding() {
  D3Plan plan;
  std::string error;
  const std::array<std::int64_t, 2> offsets{{0, 2}};
  const std::array<std::int32_t, 2> invalid{{1, 87}};
  expect(xtbloom::detail::gfn1::make_d3_plan(1, 2, offsets.data(), invalid.data(), plan, error) ==
             XTBLOOM_STATUS_INVALID_ARGUMENT,
         "D3 plan rejects unsupported elements");

  const std::array<std::int32_t, 2> numbers{{1, 8}};
  expect(xtbloom::detail::gfn1::make_d3_plan(1, 2, offsets.data(), numbers.data(), plan, error) ==
             XTBLOOM_STATUS_SUCCESS,
         "D3 plan accepts supported elements");
  std::vector<std::byte> raw(plan.workspace_size_bytes() +
                             xtbloom::detail::gfn1::kD3WorkspaceAlignment);
  D3Workspace view;
  expect(xtbloom::detail::gfn1::bind_d3_workspace(plan, raw.data() + 1, plan.workspace_size_bytes(),
                                                  view, error) == XTBLOOM_STATUS_INVALID_ARGUMENT,
         "D3 workspace binding rejects misalignment");
}

}  // namespace

int main() {
  test_plan_workspace_and_ragged_energy();
  test_reference_interpolation_layout_and_fallback();
  test_dxtb_tblite_term_goldens();
  test_cutoff_contract();
  test_gradients_and_invariance();
  test_switch_gradient();
  test_transactional_failures();
  test_invalid_plans_and_workspace_binding();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
