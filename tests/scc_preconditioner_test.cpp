#include "model/gfn2/scc_preconditioner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "model/gfn2/aes2.hpp"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/es2.hpp"
#include "model/gfn2/scc_mixer.hpp"
#include "model/gfn2/wavefunction.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

class AlignedBuffer {
 public:
  explicit AlignedBuffer(std::size_t bytes) {
    constexpr std::size_t alignment = xtbloom::detail::gfn2::kWavefunctionWorkspaceAlignment;
    bytes_ = (std::max<std::size_t>(bytes, 1u) + alignment - 1u) & ~(alignment - 1u);
    data_ = std::aligned_alloc(alignment, bytes_);
    if (data_ != nullptr) {
      std::memset(data_, 0, bytes_);
    }
  }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
  ~AlignedBuffer() { std::free(data_); }

  [[nodiscard]] void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0u;
};

struct Fixture {
  xtbloom::detail::gfn2::BasisPlan basis;
  xtbloom::detail::gfn2::WavefunctionLayout layout;
  xtbloom::detail::gfn2::ES2Plan es2;
  xtbloom::detail::gfn2::AES2Plan aes2;
  xtbloom::detail::gfn2::SccMixerPlan mixer;
  xtbloom::detail::gfn2::SccPreconditionerPlan preconditioner;
  std::unique_ptr<AlignedBuffer> wavefunction_storage;
  std::unique_ptr<AlignedBuffer> state_storage;
  std::unique_ptr<AlignedBuffer> scratch_storage;
  xtbloom::detail::gfn2::WavefunctionView wavefunction;
  xtbloom::detail::gfn2::SccMixerState state;
  xtbloom::detail::gfn2::SccMixerWorkspace scratch;
  std::array<double, 6u> positions{{-0.7, 0.0, 0.0, 0.7, 0.0, 0.0}};
  std::vector<double> es2_matrix;
  std::vector<double> es2_matrix_scratch;
  std::vector<double> es2_shell_scratch;
  std::vector<double> es2_batch_scratch;
  std::vector<double> es2_gradient_scratch;
  xtbloom::detail::gfn2::ES2Workspace es2_workspace;
  xtbloom::detail::gfn2::ES2GeometryCache es2_cache;
  std::vector<double> pair_factors;
  std::vector<double> pair_factor_scratch;
  std::vector<double> pair_constraints;
  std::vector<double> pair_constraint_scratch;
  std::vector<double> pair_denominators;
  std::vector<double> pair_denominator_scratch;
  std::vector<std::uint8_t> pair_enabled;
  std::vector<std::uint8_t> pair_enabled_scratch;
  xtbloom::detail::gfn2::SccPairResponseWorkspace pair_workspace;
  xtbloom::detail::gfn2::SccPairResponseGeometryCache pair_cache;
};

bool make_fixture(Fixture& fixture, std::string& error, double molecular_charge = 0.0,
                  std::int32_t unpaired_electrons = 0, std::int32_t spin_channel_count = 1,
                  const std::array<std::int32_t, 2u>& atomic_numbers = {{6, 8}}) {
  const std::int64_t atom_offsets[2]{0, 2};
  const double charges[1]{molecular_charge};
  const std::int32_t unpaired[1]{unpaired_electrons};
  const std::int32_t spin_channels[1]{spin_channel_count};
  if (xtbloom::detail::gfn2::make_basis_plan(1, 2, atom_offsets, atomic_numbers.data(),
                                             fixture.basis, error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_wavefunction_layout(fixture.basis, atomic_numbers.data(), charges,
                                                      unpaired, spin_channels, fixture.layout,
                                                      error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_es2_plan(fixture.basis, atomic_numbers.data(), fixture.es2,
                                           error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_aes2_plan(fixture.basis, atomic_numbers.data(), fixture.aes2,
                                            error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_scc_mixer_plan(fixture.layout, 4, 0.4, 1.0e-8, 2.0e-8,
                                                 fixture.mixer, error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_scc_preconditioner_plan(fixture.layout, fixture.es2, fixture.aes2,
                                                          fixture.preconditioner,
                                                          error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  fixture.wavefunction_storage =
      std::make_unique<AlignedBuffer>(fixture.layout.workspace_size_bytes);
  fixture.state_storage = std::make_unique<AlignedBuffer>(fixture.mixer.state_size_bytes());
  fixture.scratch_storage = std::make_unique<AlignedBuffer>(fixture.mixer.workspace_size_bytes());
  if (fixture.wavefunction_storage->data() == nullptr || fixture.state_storage->data() == nullptr ||
      fixture.scratch_storage->data() == nullptr) {
    error = "test fixture allocation failed";
    return false;
  }
  if (xtbloom::detail::gfn2::bind_wavefunction_view(
          fixture.layout, fixture.wavefunction_storage->data(),
          fixture.wavefunction_storage->size(), fixture.wavefunction,
          error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::bind_scc_mixer_state(fixture.mixer, fixture.state_storage->data(),
                                                  fixture.state_storage->size(), fixture.state,
                                                  error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::bind_scc_mixer_workspace(
          fixture.mixer, fixture.scratch_storage->data(), fixture.scratch_storage->size(),
          fixture.scratch, error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::initialize_scc_mixer_state_cpu(
          fixture.mixer, fixture.wavefunction, fixture.state, error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }

  const std::size_t matrix_elements = static_cast<std::size_t>(fixture.es2.total_matrix_elements());
  const std::size_t shell_elements = static_cast<std::size_t>(fixture.es2.total_shells());
  fixture.es2_matrix.resize(matrix_elements);
  fixture.es2_matrix_scratch.resize(matrix_elements);
  fixture.es2_shell_scratch.resize(shell_elements);
  fixture.es2_batch_scratch.resize(1u);
  fixture.es2_gradient_scratch.resize(6u);
  fixture.es2_workspace = {
      fixture.es2_matrix_scratch.data(),   fixture.es2.total_matrix_elements(),
      fixture.es2_shell_scratch.data(),    fixture.es2.total_shells(),
      fixture.es2_batch_scratch.data(),    1,
      fixture.es2_gradient_scratch.data(), 6,
  };
  if (xtbloom::detail::gfn2::update_es2_geometry_cache_cpu(
          fixture.es2, fixture.positions.data(), 1u, fixture.es2_matrix.data(),
          fixture.es2_matrix.size(), fixture.es2_workspace, fixture.es2_cache,
          error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }

  fixture.pair_factors.resize(matrix_elements);
  fixture.pair_factor_scratch.resize(matrix_elements);
  fixture.pair_constraints.resize(shell_elements);
  fixture.pair_constraint_scratch.resize(shell_elements);
  fixture.pair_denominators.resize(1u);
  fixture.pair_denominator_scratch.resize(1u);
  fixture.pair_enabled.resize(1u);
  fixture.pair_enabled_scratch.resize(1u);
  fixture.pair_workspace = {
      fixture.pair_factor_scratch.data(),      fixture.es2.total_matrix_elements(),
      fixture.pair_constraint_scratch.data(),  fixture.es2.total_shells(),
      fixture.pair_denominator_scratch.data(), 1,
      fixture.pair_enabled_scratch.data(),     1,
  };
  fixture.pair_cache = {
      fixture.pair_factors.data(),
      fixture.es2.total_matrix_elements(),
      fixture.pair_constraints.data(),
      fixture.es2.total_shells(),
      fixture.pair_denominators.data(),
      1,
      fixture.pair_enabled.data(),
      1,
      0u,
      nullptr,
  };
  return xtbloom::detail::gfn2::update_scc_pair_response_geometry_cache_cpu(
             fixture.preconditioner, fixture.es2_cache, false, fixture.pair_workspace,
             fixture.pair_cache, error) == XTBLOOM_STATUS_SUCCESS;
}

bool refresh_pair_cache(Fixture& fixture, bool periodic_response_enabled, std::string& error) {
  return xtbloom::detail::gfn2::update_scc_pair_response_geometry_cache_cpu(
             fixture.preconditioner, fixture.es2_cache, periodic_response_enabled,
             fixture.pair_workspace, fixture.pair_cache, error) == XTBLOOM_STATUS_SUCCESS;
}

void install_residual(Fixture& fixture, const std::vector<double>& residual) {
  std::size_t packed = 0u;
  const std::size_t vector_begin = static_cast<std::size_t>(fixture.mixer.vector_offsets().front());
  const std::array<double*, 3u> fields{
      {fixture.wavefunction.qsh, fixture.wavefunction.dipole, fixture.wavefunction.quadrupole}};
  const std::array<const xtbloom::detail::gfn2::WavefunctionFieldLayout*, 3u> layouts{
      {&fixture.layout.qsh, &fixture.layout.dipole, &fixture.layout.quadrupole}};
  for (std::size_t field = 0u; field < fields.size(); ++field) {
    const std::size_t begin = static_cast<std::size_t>(layouts[field]->system_offsets[0]);
    const std::size_t end = static_cast<std::size_t>(layouts[field]->system_offsets[1]);
    for (std::size_t destination = begin; destination < end; ++destination, ++packed) {
      fields[field][destination] =
          fixture.state.current_inputs[vector_begin + packed] + residual[packed];
    }
  }
}

bool solve_dense_long_double(std::vector<long double>& matrix, std::vector<long double>& values,
                             std::size_t dimension) {
  for (std::size_t column = 0u; column < dimension; ++column) {
    std::size_t pivot = column;
    for (std::size_t row = column + 1u; row < dimension; ++row) {
      if (std::abs(matrix[row * dimension + column]) >
          std::abs(matrix[pivot * dimension + column])) {
        pivot = row;
      }
    }
    if (matrix[pivot * dimension + column] == 0.0L) {
      return false;
    }
    if (pivot != column) {
      for (std::size_t entry = column; entry < dimension; ++entry) {
        std::swap(matrix[column * dimension + entry], matrix[pivot * dimension + entry]);
      }
      std::swap(values[column], values[pivot]);
    }
    for (std::size_t row = column + 1u; row < dimension; ++row) {
      const long double scale =
          matrix[row * dimension + column] / matrix[column * dimension + column];
      matrix[row * dimension + column] = 0.0L;
      for (std::size_t entry = column + 1u; entry < dimension; ++entry) {
        matrix[row * dimension + entry] -= scale * matrix[column * dimension + entry];
      }
      values[row] -= scale * values[column];
    }
  }
  for (std::size_t reverse = dimension; reverse > 0u; --reverse) {
    const std::size_t row = reverse - 1u;
    long double value = values[row];
    for (std::size_t column = row + 1u; column < dimension; ++column) {
      value -= matrix[row * dimension + column] * values[column];
    }
    values[row] = value / matrix[row * dimension + row];
  }
  return true;
}

std::vector<long double> independent_pair_kkt_reference(const Fixture& fixture,
                                                        const std::vector<double>& residual) {
  const std::size_t shells = static_cast<std::size_t>(fixture.es2.total_shells());
  const std::size_t dimension = shells + 1u;
  std::vector<long double> matrix(dimension * dimension, 0.0L);
  std::vector<long double> right_hand_side(dimension, 0.0L);
  for (std::size_t row = 0u; row < shells; ++row) {
    const long double diagonal = static_cast<long double>(fixture.es2.shell_hardness()[row]) /
                                 xtbloom::detail::gfn2::kSccPairResponseScale;
    right_hand_side[row] = diagonal * static_cast<long double>(residual[row]);
    for (std::size_t column = 0u; column < shells; ++column) {
      if (row == column) {
        matrix[row * dimension + column] = diagonal;
      } else if (fixture.es2.shell_to_atom()[row] != fixture.es2.shell_to_atom()[column]) {
        matrix[row * dimension + column] =
            static_cast<long double>(fixture.es2_matrix[row * shells + column]);
      }
    }
    matrix[row * dimension + shells] = 1.0L;
    matrix[shells * dimension + row] = 1.0L;
  }
  if (!solve_dense_long_double(matrix, right_hand_side, dimension)) {
    return {};
  }
  right_hand_side.resize(shells);
  return right_hand_side;
}

int test_metric_and_local_preconditioner() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(fixture, error));
  CHECK(fixture.preconditioner.sealed());
  CHECK(fixture.preconditioner.vector_offsets() == fixture.mixer.vector_offsets());
  CHECK(std::all_of(fixture.preconditioner.charge_shell_scales().begin(),
                    fixture.preconditioner.charge_shell_scales().end(),
                    [](double value) { return value >= 0.5 && value <= 2.0; }));

  const std::size_t q_count = static_cast<std::size_t>(fixture.layout.qsh.element_count);
  const std::size_t d_count = static_cast<std::size_t>(fixture.layout.dipole.element_count);
  const std::size_t dimension = static_cast<std::size_t>(fixture.mixer.total_vector_elements());
  std::vector<double> residual(dimension, 0.0);
  double q_sum = 0.0;
  for (std::size_t shell = 0u; shell + 1u < q_count; ++shell) {
    residual[shell] = 0.01 * static_cast<double>(shell + 1u);
    q_sum += residual[shell];
  }
  residual[q_count - 1u] = -q_sum;
  for (std::size_t component = q_count; component < dimension; ++component) {
    residual[component] = 0.001 * static_cast<double>(component + 1u);
  }
  install_residual(fixture, residual);

  xtbloom::detail::gfn2::SccResidualDiagnostics diagnostics;
  CHECK(xtbloom::detail::gfn2::prepare_scc_residual_system_cpu(
            fixture.preconditioner, xtbloom::detail::gfn2::SccResidualPolicy::kControllerOnly, 0,
            fixture.wavefunction, fixture.pair_cache, fixture.state, fixture.scratch, diagnostics,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::equal(residual.begin(), residual.end(), fixture.scratch.residual));
  double raw_square = 0.0;
  double raw_maximum = 0.0;
  for (double value : residual) {
    raw_square += value * value;
    raw_maximum = std::max(raw_maximum, std::abs(value));
  }
  CHECK(diagnostics.raw_residual_rms ==
        std::sqrt(raw_square) / std::sqrt(static_cast<double>(dimension)));
  CHECK(diagnostics.raw_residual_maximum == raw_maximum);
  CHECK(!diagnostics.has_previous_residual);

  const auto& weights = fixture.preconditioner.metric_weights();
  const std::size_t quadrupole_begin = q_count + d_count;
  CHECK(weights[quadrupole_begin + 1u] == 2.0 * weights[quadrupole_begin]);
  CHECK(weights[quadrupole_begin + 2u] == weights[quadrupole_begin]);
  CHECK(weights[quadrupole_begin + 3u] == 2.0 * weights[quadrupole_begin]);
  CHECK(weights[quadrupole_begin + 4u] == 2.0 * weights[quadrupole_begin]);
  CHECK(weights[quadrupole_begin + 5u] == weights[quadrupole_begin]);

  std::vector<double> identity_effective(fixture.scratch.residual,
                                         fixture.scratch.residual + dimension);
  CHECK(xtbloom::detail::gfn2::prepare_scc_residual_system_cpu(
            fixture.preconditioner, xtbloom::detail::gfn2::SccResidualPolicy::kLocalV1, 0,
            fixture.wavefunction, fixture.pair_cache, fixture.state, fixture.scratch, diagnostics,
            error) == XTBLOOM_STATUS_SUCCESS);
  double effective_q_sum = 0.0;
  bool charge_changed = false;
  for (std::size_t shell = 0u; shell < q_count; ++shell) {
    effective_q_sum += fixture.scratch.residual[shell];
    charge_changed = charge_changed || fixture.scratch.residual[shell] != residual[shell];
  }
  CHECK(std::abs(effective_q_sum) < 1.0e-15);
  CHECK(charge_changed);
  CHECK(std::equal(identity_effective.begin() + static_cast<std::ptrdiff_t>(q_count),
                   identity_effective.end(), fixture.scratch.residual + q_count));

  std::fill(residual.begin(), residual.end(), 0.0);
  for (std::size_t shell = 0u; shell < q_count; ++shell) {
    residual[shell] = 0.002 * static_cast<double>(shell + 1u);
  }
  install_residual(fixture, residual);
  CHECK(xtbloom::detail::gfn2::prepare_scc_residual_system_cpu(
            fixture.preconditioner, xtbloom::detail::gfn2::SccResidualPolicy::kLocalV1, 0,
            fixture.wavefunction, fixture.pair_cache, fixture.state, fixture.scratch, diagnostics,
            error) == XTBLOOM_STATUS_SUCCESS);
  double raw_nonzero_sum = 0.0;
  double effective_nonzero_sum = 0.0;
  for (std::size_t shell = 0u; shell < q_count; ++shell) {
    raw_nonzero_sum += residual[shell];
    effective_nonzero_sum += fixture.scratch.residual[shell];
  }
  CHECK(std::abs(effective_nonzero_sum - raw_nonzero_sum) < 1.0e-15);
  return 0;
}

int test_previous_weighted_angle_uses_effective_history() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(fixture, error));
  const std::size_t dimension = static_cast<std::size_t>(fixture.mixer.total_vector_elements());
  std::vector<double> residual(dimension, 0.001);
  install_residual(fixture, residual);
  xtbloom::detail::gfn2::SccResidualDiagnostics diagnostics;
  CHECK(xtbloom::detail::gfn2::prepare_scc_residual_system_cpu(
            fixture.preconditioner, xtbloom::detail::gfn2::SccResidualPolicy::kControllerOnly, 0,
            fixture.wavefunction, fixture.pair_cache, fixture.state, fixture.scratch, diagnostics,
            error) == XTBLOOM_STATUS_SUCCESS);
  const xtbloom::detail::gfn2::SccMixerPreparedStepView prepared{
      fixture.scratch.residual,
      dimension,
      diagnostics.raw_residual_rms,
      diagnostics.raw_residual_maximum,
      0.4,
      false,
      fixture.preconditioner.metric_weights().data(),
      std::numeric_limits<double>::max()};
  CHECK(xtbloom::detail::gfn2::mix_scc_broyden_system_cpu_prepared(
            fixture.mixer, 0, fixture.wavefunction, fixture.state, fixture.scratch, prepared,
            error) == XTBLOOM_STATUS_SUCCESS);
  install_residual(fixture, residual);
  CHECK(xtbloom::detail::gfn2::prepare_scc_residual_system_cpu(
            fixture.preconditioner, xtbloom::detail::gfn2::SccResidualPolicy::kControllerOnly, 0,
            fixture.wavefunction, fixture.pair_cache, fixture.state, fixture.scratch, diagnostics,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(diagnostics.has_previous_residual);
  CHECK(diagnostics.cosine_is_valid);
  CHECK(std::abs(diagnostics.weighted_residual_cosine - 1.0) < 1.0e-15);
  CHECK(diagnostics.weighted_residual_norm == diagnostics.previous_weighted_residual_norm);
  return 0;
}

int test_unrestricted_channel_sums_and_initial_magnetization_are_preserved() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(fixture, error, 1.0, 1, 2));
  const std::size_t q_count = static_cast<std::size_t>(fixture.layout.qsh.element_count);
  const std::size_t shell_count = q_count / 2u;
  const std::size_t dimension = static_cast<std::size_t>(fixture.mixer.total_vector_elements());
  std::vector<double> residual(dimension, 0.0);
  for (std::size_t shell = 0u; shell < shell_count; ++shell) {
    residual[shell] = 0.001 * static_cast<double>(shell + 1u);
    residual[shell_count + shell] = 0.002 * static_cast<double>(shell + 1u);
  }
  install_residual(fixture, residual);

  xtbloom::detail::gfn2::SccResidualDiagnostics diagnostics;
  CHECK(xtbloom::detail::gfn2::prepare_scc_residual_system_cpu(
            fixture.preconditioner, xtbloom::detail::gfn2::SccResidualPolicy::kLocalV1, 0,
            fixture.wavefunction, fixture.pair_cache, fixture.state, fixture.scratch, diagnostics,
            error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t channel = 0u; channel < 2u; ++channel) {
    double raw_sum = 0.0;
    double effective_sum = 0.0;
    for (std::size_t shell = 0u; shell < shell_count; ++shell) {
      const std::size_t component = channel * shell_count + shell;
      raw_sum += residual[component];
      effective_sum += fixture.scratch.residual[component];
      if (channel == 1u) {
        CHECK(fixture.scratch.residual[component] == residual[component]);
      }
    }
    CHECK(std::abs(effective_sum - raw_sum) < 1.0e-15);
  }

  Fixture pair_fixture;
  CHECK(make_fixture(pair_fixture, error, 0.0, 0, 2));
  const std::size_t pair_q_count = static_cast<std::size_t>(pair_fixture.layout.qsh.element_count);
  const std::size_t pair_shell_count = pair_q_count / 2u;
  const std::size_t pair_dimension =
      static_cast<std::size_t>(pair_fixture.mixer.total_vector_elements());
  std::vector<double> pair_residual(pair_dimension, 0.0);
  double first_channel_sum = 0.0;
  for (std::size_t shell = 0u; shell + 1u < pair_shell_count; ++shell) {
    pair_residual[shell] = 0.003 * static_cast<double>(shell + 1u);
    first_channel_sum += pair_residual[shell];
  }
  pair_residual[pair_shell_count - 1u] = -first_channel_sum;
  for (std::size_t shell = 0u; shell < pair_shell_count; ++shell) {
    pair_residual[pair_shell_count + shell] = 0.004 * static_cast<double>(shell + 1u);
  }
  for (std::size_t component = pair_q_count; component < pair_dimension; ++component) {
    pair_residual[component] = 0.0001 * static_cast<double>(component + 1u);
  }
  install_residual(pair_fixture, pair_residual);
  CHECK(xtbloom::detail::gfn2::prepare_scc_residual_system_cpu(
            pair_fixture.preconditioner, xtbloom::detail::gfn2::SccResidualPolicy::kPairResponseV1,
            0, pair_fixture.wavefunction, pair_fixture.pair_cache, pair_fixture.state,
            pair_fixture.scratch, diagnostics, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(diagnostics.preconditioner_applied);
  CHECK(!diagnostics.preconditioner_fell_back);
  CHECK(std::equal(pair_residual.begin() + static_cast<std::ptrdiff_t>(pair_shell_count),
                   pair_residual.end(), pair_fixture.scratch.residual + pair_shell_count));
  return 0;
}

int test_pair_response_matches_two_shell_constrained_reference() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(fixture, error, 0.0, 0, 1, {{1, 1}}));
  CHECK(fixture.es2.total_shells() == 2);
  CHECK(fixture.pair_enabled == std::vector<std::uint8_t>{1u});

  const std::size_t dimension = static_cast<std::size_t>(fixture.mixer.total_vector_elements());
  std::vector<double> residual(dimension, 0.0);
  residual[0] = 0.02;
  residual[1] = -0.02;
  for (std::size_t component = 2u; component < dimension; ++component) {
    residual[component] = 0.0002 * static_cast<double>(component + 1u);
  }
  install_residual(fixture, residual);

  const std::vector<double> factor_before = fixture.pair_factors;
  const std::vector<double> constraint_before = fixture.pair_constraints;
  const std::vector<double> denominator_before = fixture.pair_denominators;
  xtbloom::detail::gfn2::SccResidualDiagnostics diagnostics;
  CHECK(xtbloom::detail::gfn2::prepare_scc_residual_system_cpu(
            fixture.preconditioner, xtbloom::detail::gfn2::SccResidualPolicy::kPairResponseV1, 0,
            fixture.wavefunction, fixture.pair_cache, fixture.state, fixture.scratch, diagnostics,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(diagnostics.preconditioner_applied);
  CHECK(!diagnostics.preconditioner_fell_back);

  const double hardness = fixture.es2.shell_hardness()[0];
  CHECK(hardness == fixture.es2.shell_hardness()[1]);
  const double diagonal = hardness / xtbloom::detail::gfn2::kSccPairResponseScale;
  const double pair_coupling = fixture.es2_matrix[1];
  const double expected_factor = diagonal / (diagonal - pair_coupling);
  const double tolerance = 32.0 * std::numeric_limits<double>::epsilon() * expected_factor;
  CHECK(std::abs(fixture.scratch.residual[0] - expected_factor * residual[0]) < tolerance);
  CHECK(std::abs(fixture.scratch.residual[1] - expected_factor * residual[1]) < tolerance);
  CHECK(fixture.scratch.residual[0] + fixture.scratch.residual[1] == 0.0);
  CHECK(std::equal(residual.begin() + 2, residual.end(), fixture.scratch.residual + 2));
  CHECK(fixture.pair_factors == factor_before);
  CHECK(fixture.pair_constraints == constraint_before);
  CHECK(fixture.pair_denominators == denominator_before);

  residual[0] = -0.02;
  residual[1] = 0.02;
  install_residual(fixture, residual);
  CHECK(xtbloom::detail::gfn2::prepare_scc_residual_system_cpu(
            fixture.preconditioner, xtbloom::detail::gfn2::SccResidualPolicy::kPairResponseV1, 0,
            fixture.wavefunction, fixture.pair_cache, fixture.state, fixture.scratch, diagnostics,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::abs(fixture.scratch.residual[0] - expected_factor * residual[0]) < tolerance);
  CHECK(std::abs(fixture.scratch.residual[1] - expected_factor * residual[1]) < tolerance);

  fixture.positions[0] = -1.4;
  fixture.positions[3] = 1.4;
  CHECK(xtbloom::detail::gfn2::update_es2_geometry_cache_cpu(
            fixture.es2, fixture.positions.data(), 2u, fixture.es2_matrix.data(),
            fixture.es2_matrix.size(), fixture.es2_workspace, fixture.es2_cache,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(refresh_pair_cache(fixture, false, error));
  CHECK(fixture.pair_cache.geometry_generation == 2u);
  CHECK(fixture.pair_factors != factor_before);
  const double refreshed_coupling = fixture.es2_matrix[1];
  const double refreshed_factor = diagonal / (diagonal - refreshed_coupling);
  residual[0] = 0.02;
  residual[1] = -0.02;
  install_residual(fixture, residual);
  CHECK(xtbloom::detail::gfn2::prepare_scc_residual_system_cpu(
            fixture.preconditioner, xtbloom::detail::gfn2::SccResidualPolicy::kPairResponseV1, 0,
            fixture.wavefunction, fixture.pair_cache, fixture.state, fixture.scratch, diagnostics,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::abs(fixture.scratch.residual[0] - refreshed_factor * residual[0]) < tolerance);
  CHECK(std::abs(fixture.scratch.residual[1] - refreshed_factor * residual[1]) < tolerance);
  return 0;
}

int test_pair_response_matches_dense_heterogeneous_kkt_and_permutation() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(fixture, error));
  const std::size_t shells = static_cast<std::size_t>(fixture.es2.total_shells());
  CHECK(shells > 2u);
  const std::size_t dimension = static_cast<std::size_t>(fixture.mixer.total_vector_elements());
  std::vector<double> residual(dimension, 0.0);
  double prefix = 0.0;
  for (std::size_t shell = 0u; shell + 1u < shells; ++shell) {
    residual[shell] = (shell % 2u == 0u ? 1.0 : -0.5) * 0.003 * static_cast<double>(shell + 1u);
    prefix += residual[shell];
  }
  residual[shells - 1u] = -prefix;
  for (std::size_t component = shells; component < dimension; ++component) {
    residual[component] = 0.0001 * static_cast<double>(component + 1u);
  }
  const std::vector<long double> reference = independent_pair_kkt_reference(fixture, residual);
  CHECK(reference.size() == shells);
  install_residual(fixture, residual);
  xtbloom::detail::gfn2::SccResidualDiagnostics diagnostics;
  CHECK(xtbloom::detail::gfn2::prepare_scc_residual_system_cpu(
            fixture.preconditioner, xtbloom::detail::gfn2::SccResidualPolicy::kPairResponseV1, 0,
            fixture.wavefunction, fixture.pair_cache, fixture.state, fixture.scratch, diagnostics,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(diagnostics.preconditioner_applied);
  double response_sum = 0.0;
  for (std::size_t shell = 0u; shell < shells; ++shell) {
    const double expected = static_cast<double>(reference[shell]);
    const double tolerance =
        512.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(expected));
    CHECK(std::abs(fixture.scratch.residual[shell] - expected) < tolerance);
    response_sum += fixture.scratch.residual[shell];
  }
  CHECK(response_sum == 0.0);
  CHECK(std::equal(residual.begin() + static_cast<std::ptrdiff_t>(shells), residual.end(),
                   fixture.scratch.residual + shells));

  const std::size_t first_atom_shells = static_cast<std::size_t>(
      fixture.basis.atom_shell_offsets[1] - fixture.basis.atom_shell_offsets[0]);
  const std::size_t second_atom_shells = static_cast<std::size_t>(
      fixture.basis.atom_shell_offsets[2] - fixture.basis.atom_shell_offsets[1]);
  CHECK(first_atom_shells == second_atom_shells);
  Fixture permuted;
  CHECK(make_fixture(permuted, error, 0.0, 0, 1, {{8, 6}}));
  permuted.positions = {{0.7, 0.0, 0.0, -0.7, 0.0, 0.0}};
  CHECK(xtbloom::detail::gfn2::update_es2_geometry_cache_cpu(
            permuted.es2, permuted.positions.data(), 2u, permuted.es2_matrix.data(),
            permuted.es2_matrix.size(), permuted.es2_workspace, permuted.es2_cache,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(refresh_pair_cache(permuted, false, error));
  std::vector<double> permuted_residual(dimension, 0.0);
  std::copy_n(residual.data() + first_atom_shells, second_atom_shells, permuted_residual.data());
  std::copy_n(residual.data(), first_atom_shells, permuted_residual.data() + second_atom_shells);
  install_residual(permuted, permuted_residual);
  CHECK(xtbloom::detail::gfn2::prepare_scc_residual_system_cpu(
            permuted.preconditioner, xtbloom::detail::gfn2::SccResidualPolicy::kPairResponseV1, 0,
            permuted.wavefunction, permuted.pair_cache, permuted.state, permuted.scratch,
            diagnostics, error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t shell = 0u; shell < first_atom_shells; ++shell) {
    const double expected = fixture.scratch.residual[shell];
    CHECK(std::abs(permuted.scratch.residual[second_atom_shells + shell] - expected) <
          512.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(expected)));
  }
  for (std::size_t shell = 0u; shell < second_atom_shells; ++shell) {
    const double expected = fixture.scratch.residual[first_atom_shells + shell];
    CHECK(std::abs(permuted.scratch.residual[shell] - expected) <
          512.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(expected)));
  }
  return 0;
}

int test_pair_response_cache_excludes_same_atom_blocks_and_is_atomic() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(fixture, error));
  CHECK(fixture.basis.atom_shell_offsets[1] - fixture.basis.atom_shell_offsets[0] >= 2);
  const std::size_t shell_count = static_cast<std::size_t>(fixture.es2.total_shells());
  /* The first atom is the leading Cholesky block. Because its direct same-atom
   * K_pair entries are exactly zero, no preceding atom can introduce fill. */
  CHECK(fixture.pair_factors[shell_count] == 0.0);

  const std::vector<double> factors_before = fixture.pair_factors;
  const std::vector<double> constraints_before = fixture.pair_constraints;
  const std::vector<double> denominators_before = fixture.pair_denominators;
  const std::vector<std::uint8_t> enabled_before = fixture.pair_enabled;
  const std::vector<double> matrix_before = fixture.es2_matrix;
  for (std::size_t row = 0u; row < shell_count; ++row) {
    for (std::size_t column = row; column < shell_count; ++column) {
      if (fixture.es2.shell_to_atom()[row] == fixture.es2.shell_to_atom()[column]) {
        const double sentinel = 0.25 + 0.01 * static_cast<double>(row + column);
        fixture.es2_matrix[row * shell_count + column] = sentinel;
        fixture.es2_matrix[column * shell_count + row] = sentinel;
      }
    }
  }
  CHECK(refresh_pair_cache(fixture, false, error));
  CHECK(fixture.pair_factors == factors_before);
  CHECK(fixture.pair_constraints == constraints_before);
  CHECK(fixture.pair_denominators == denominators_before);
  CHECK(fixture.pair_enabled == enabled_before);
  std::copy(matrix_before.begin(), matrix_before.end(), fixture.es2_matrix.begin());

  xtbloom::detail::gfn2::SccPairResponseWorkspace aliased = fixture.pair_workspace;
  aliased.factor_scratch = fixture.pair_cache.cholesky_factors;
  CHECK(xtbloom::detail::gfn2::update_scc_pair_response_geometry_cache_cpu(
            fixture.preconditioner, fixture.es2_cache, false, aliased, fixture.pair_cache, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.pair_factors == factors_before);
  CHECK(fixture.pair_constraints == constraints_before);
  CHECK(fixture.pair_denominators == denominators_before);
  CHECK(fixture.pair_enabled == enabled_before);

  const xtbloom::detail::gfn2::SccPairResponseGeometryCache descriptor_before = fixture.pair_cache;
  aliased = fixture.pair_workspace;
  aliased.factor_scratch = reinterpret_cast<double*>(&fixture.pair_cache);
  CHECK(xtbloom::detail::gfn2::update_scc_pair_response_geometry_cache_cpu(
            fixture.preconditioner, fixture.es2_cache, false, aliased, fixture.pair_cache, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::memcmp(&fixture.pair_cache, &descriptor_before, sizeof(descriptor_before)) == 0);
  CHECK(fixture.pair_factors == factors_before);
  CHECK(fixture.pair_constraints == constraints_before);
  CHECK(fixture.pair_denominators == denominators_before);
  CHECK(fixture.pair_enabled == enabled_before);

  CHECK(refresh_pair_cache(fixture, false, error));
  CHECK(fixture.pair_factors == factors_before);
  CHECK(fixture.pair_constraints == constraints_before);
  CHECK(fixture.pair_denominators == denominators_before);
  CHECK(fixture.pair_enabled == enabled_before);
  return 0;
}

int test_pair_response_falls_back_without_modifying_the_raw_residual() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(fixture, error, 0.0, 0, 1, {{1, 1}}));
  const std::size_t dimension = static_cast<std::size_t>(fixture.mixer.total_vector_elements());
  std::vector<double> residual(dimension, 0.0);
  residual[0] = 0.01;
  residual[1] = -0.01;
  for (std::size_t component = 2u; component < dimension; ++component) {
    residual[component] = 0.0003 * static_cast<double>(component + 1u);
  }
  xtbloom::detail::gfn2::SccResidualDiagnostics diagnostics;
  const auto expect_raw_fallback = [&]() {
    install_residual(fixture, residual);
    if (xtbloom::detail::gfn2::prepare_scc_residual_system_cpu(
            fixture.preconditioner, xtbloom::detail::gfn2::SccResidualPolicy::kPairResponseV1, 0,
            fixture.wavefunction, fixture.pair_cache, fixture.state, fixture.scratch, diagnostics,
            error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    return diagnostics.preconditioner_fell_back && !diagnostics.preconditioner_applied &&
           std::equal(residual.begin(), residual.end(), fixture.scratch.residual);
  };

  residual[1] = 0.0;
  CHECK(expect_raw_fallback());
  residual[1] = -residual[0];

  const std::vector<double> valid_matrix = fixture.es2_matrix;
  fixture.es2_matrix[1] = std::numeric_limits<double>::quiet_NaN();
  fixture.es2_matrix[2] = std::numeric_limits<double>::quiet_NaN();
  CHECK(refresh_pair_cache(fixture, false, error));
  CHECK(fixture.pair_enabled[0] == 0u);
  CHECK(expect_raw_fallback());

  std::copy(valid_matrix.begin(), valid_matrix.end(), fixture.es2_matrix.begin());
  CHECK(refresh_pair_cache(fixture, false, error));
  const double diagonal =
      fixture.es2.shell_hardness()[0] / xtbloom::detail::gfn2::kSccPairResponseScale;
  fixture.es2_matrix[1] = 2.0 * diagonal;
  fixture.es2_matrix[2] = 2.0 * diagonal;
  CHECK(refresh_pair_cache(fixture, false, error));
  CHECK(fixture.pair_enabled[0] == 0u);
  CHECK(expect_raw_fallback());

  std::copy(valid_matrix.begin(), valid_matrix.end(), fixture.es2_matrix.begin());
  const double conditioning_boundary = std::sqrt(std::numeric_limits<double>::epsilon());
  fixture.es2_matrix[1] = diagonal * (1.0 - 0.5 * conditioning_boundary);
  fixture.es2_matrix[2] = fixture.es2_matrix[1];
  CHECK(refresh_pair_cache(fixture, false, error));
  CHECK(fixture.pair_enabled[0] == 0u);
  CHECK(expect_raw_fallback());

  std::copy(valid_matrix.begin(), valid_matrix.end(), fixture.es2_matrix.begin());
  CHECK(refresh_pair_cache(fixture, true, error));
  CHECK(fixture.pair_enabled[0] == 0u);
  CHECK(expect_raw_fallback());

  CHECK(refresh_pair_cache(fixture, false, error));
  fixture.pair_denominators[0] = 0.0;
  CHECK(expect_raw_fallback());
  CHECK(refresh_pair_cache(fixture, false, error));
  fixture.pair_denominators[0] = std::numeric_limits<double>::min();
  CHECK(expect_raw_fallback());
  CHECK(refresh_pair_cache(fixture, false, error));
  fixture.pair_constraints[0] *= 1.0 + 1.0e-10;
  CHECK(expect_raw_fallback());
  return 0;
}

int test_weighted_norm_and_angle_are_rotation_invariant() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(fixture, error));
  const std::size_t q_count = static_cast<std::size_t>(fixture.layout.qsh.element_count);
  const std::size_t d_count = static_cast<std::size_t>(fixture.layout.dipole.element_count);
  const std::size_t dimension = static_cast<std::size_t>(fixture.mixer.total_vector_elements());
  const std::size_t dipole_begin = q_count;
  const std::size_t quadrupole_begin = q_count + d_count;

  std::vector<double> current(dimension, 0.0);
  current[dipole_begin + 0u] = 0.001;
  current[dipole_begin + 1u] = 0.002;
  current[dipole_begin + 2u] = 0.003;
  current[quadrupole_begin + 0u] = 0.0003;
  current[quadrupole_begin + 1u] = 0.0002;
  current[quadrupole_begin + 2u] = -0.0004;
  current[quadrupole_begin + 3u] = -0.0001;
  current[quadrupole_begin + 4u] = 0.00005;
  current[quadrupole_begin + 5u] = 0.0001;
  std::vector<double> previous(dimension, 0.0);
  previous[dipole_begin + 0u] = 0.0002;
  previous[dipole_begin + 1u] = -0.0004;
  previous[dipole_begin + 2u] = 0.0007;

  fixture.state.history_ages[0] = 1u;
  std::copy(previous.begin(), previous.end(), fixture.state.previous_residuals);
  install_residual(fixture, current);
  xtbloom::detail::gfn2::SccResidualDiagnostics original;
  CHECK(xtbloom::detail::gfn2::prepare_scc_residual_system_cpu(
            fixture.preconditioner, xtbloom::detail::gfn2::SccResidualPolicy::kControllerOnly, 0,
            fixture.wavefunction, fixture.pair_cache, fixture.state, fixture.scratch, original,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(original.cosine_is_valid);

  /* Rotate both tensors by +90 degrees about z. Packed Q uses
   * [xx,xy,yy,xz,yz,zz], so R Q R^T maps as written below. */
  std::vector<double> rotated_current = current;
  rotated_current[dipole_begin + 0u] = -current[dipole_begin + 1u];
  rotated_current[dipole_begin + 1u] = current[dipole_begin + 0u];
  rotated_current[quadrupole_begin + 0u] = current[quadrupole_begin + 2u];
  rotated_current[quadrupole_begin + 1u] = -current[quadrupole_begin + 1u];
  rotated_current[quadrupole_begin + 2u] = current[quadrupole_begin + 0u];
  rotated_current[quadrupole_begin + 3u] = -current[quadrupole_begin + 4u];
  rotated_current[quadrupole_begin + 4u] = current[quadrupole_begin + 3u];
  std::vector<double> rotated_previous = previous;
  rotated_previous[dipole_begin + 0u] = -previous[dipole_begin + 1u];
  rotated_previous[dipole_begin + 1u] = previous[dipole_begin + 0u];
  std::copy(rotated_previous.begin(), rotated_previous.end(), fixture.state.previous_residuals);
  install_residual(fixture, rotated_current);
  xtbloom::detail::gfn2::SccResidualDiagnostics rotated;
  CHECK(xtbloom::detail::gfn2::prepare_scc_residual_system_cpu(
            fixture.preconditioner, xtbloom::detail::gfn2::SccResidualPolicy::kControllerOnly, 0,
            fixture.wavefunction, fixture.pair_cache, fixture.state, fixture.scratch, rotated,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(rotated.cosine_is_valid);
  CHECK(std::abs(rotated.weighted_residual_norm - original.weighted_residual_norm) < 1.0e-15);
  CHECK(std::abs(rotated.previous_weighted_residual_norm -
                 original.previous_weighted_residual_norm) < 1.0e-15);
  CHECK(std::abs(rotated.weighted_residual_cosine - original.weighted_residual_cosine) < 1.0e-15);
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_metric_and_local_preconditioner(); status != 0) {
    return status;
  }
  if (const int status = test_previous_weighted_angle_uses_effective_history(); status != 0) {
    return status;
  }
  if (const int status = test_unrestricted_channel_sums_and_initial_magnetization_are_preserved();
      status != 0) {
    return status;
  }
  if (const int status = test_pair_response_matches_two_shell_constrained_reference();
      status != 0) {
    return status;
  }
  if (const int status = test_pair_response_matches_dense_heterogeneous_kkt_and_permutation();
      status != 0) {
    return status;
  }
  if (const int status = test_pair_response_cache_excludes_same_atom_blocks_and_is_atomic();
      status != 0) {
    return status;
  }
  if (const int status = test_pair_response_falls_back_without_modifying_the_raw_residual();
      status != 0) {
    return status;
  }
  return test_weighted_norm_and_angle_are_rotation_invariant();
}
