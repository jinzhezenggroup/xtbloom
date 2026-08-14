#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "model/common/scc_mixer.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

using xtbloom::detail::common::SccMixerPlan;
using xtbloom::detail::common::SccMixerPreparedStepView;
using xtbloom::detail::common::SccMixerState;
using xtbloom::detail::common::SccMixerVectorLayoutView;
using xtbloom::detail::common::SccMixerVectorView;
using xtbloom::detail::common::SccMixerWorkspace;

class AlignedBuffer {
 public:
  explicit AlignedBuffer(std::size_t bytes) {
    constexpr std::size_t alignment = xtbloom::detail::common::kSccMixerWorkspaceAlignment;
    bytes_ = (std::max<std::size_t>(bytes, 1u) + alignment - 1u) & ~(alignment - 1u);
    data_ = std::aligned_alloc(xtbloom::detail::common::kSccMixerWorkspaceAlignment, bytes_);
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
  static constexpr std::size_t kVectorBytes = 128u;
  static constexpr std::size_t kQshOffset = 64u;
  static constexpr std::size_t kQshElements = 6u;
  /* Keep byte-address arithmetic explicit: this is the first guard byte after
   * the double-valued qsh field, not an element offset on a typed pointer. */
  static constexpr std::size_t kQshEndOffset = kQshOffset + kQshElements * sizeof(double);
  static constexpr std::size_t kSuffixBytes = kVectorBytes - kQshEndOffset;

  std::int64_t qsh_offsets[3]{0, 2, 6};
  std::array<double, 4u> unit_metric{{1.0, 1.0, 1.0, 1.0}};
  SccMixerVectorLayoutView layout;
  SccMixerPlan plan;
  AlignedBuffer vector_storage{kVectorBytes};
  std::unique_ptr<AlignedBuffer> state_storage;
  std::unique_ptr<AlignedBuffer> scratch_storage;
  SccMixerVectorView vector;
  SccMixerState state;
  SccMixerWorkspace scratch;

  explicit Fixture(std::string& error) {
    make_plan(error);
    if (!plan.sealed()) {
      return;
    }
    state_storage = std::make_unique<AlignedBuffer>(plan.state_size_bytes());
    scratch_storage = std::make_unique<AlignedBuffer>(plan.workspace_size_bytes());
    if (vector_storage.data() == nullptr || state_storage->data() == nullptr ||
        scratch_storage->data() == nullptr) {
      return;
    }
    vector.workspace_base = vector_storage.data();
    vector.workspace_size_bytes = vector_storage.size();
    vector.fields[0] =
        reinterpret_cast<double*>(static_cast<std::byte*>(vector_storage.data()) + kQshOffset);
    vector.field_count = 1u;
    if (xtbloom::detail::common::bind_scc_mixer_state(plan, state_storage->data(),
                                                      state_storage->size(), state,
                                                      error) != XTBLOOM_STATUS_SUCCESS ||
        xtbloom::detail::common::bind_scc_mixer_workspace(plan, scratch_storage->data(),
                                                          scratch_storage->size(), scratch,
                                                          error) != XTBLOOM_STATUS_SUCCESS) {
      plan = {};
    }
  }

 private:
  void make_plan(std::string& error) {
    layout.batch_size = 2;
    layout.workspace_size_bytes = kVectorBytes;
    layout.workspace_alignment = xtbloom::detail::common::kSccMixerWorkspaceAlignment;
    layout.field_count = 1u;
    layout.fields[0] = {kQshOffset, kQshElements * sizeof(double),
                        static_cast<std::int64_t>(kQshElements), qsh_offsets, 3u};
    if (xtbloom::detail::common::make_scc_mixer_plan(layout, 3, 0.4, 1.0e-8, 2.0e-8, plan, error) !=
        XTBLOOM_STATUS_SUCCESS) {
      return;
    }
  }
};

bool equal(const double* first, const double* second, std::size_t count) {
  return std::equal(first, first + count, second);
}

SccMixerPreparedStepView prepare_raw_residual(Fixture& fixture, std::size_t system, double damping,
                                              bool restart_history = false) {
  const std::size_t begin = static_cast<std::size_t>(fixture.plan.vector_offsets()[system]);
  const std::size_t end = static_cast<std::size_t>(fixture.plan.vector_offsets()[system + 1u]);
  const std::size_t dimension = end - begin;
  const std::size_t field_begin = static_cast<std::size_t>(fixture.qsh_offsets[system]);
  double square = 0.0;
  double maximum = 0.0;
  for (std::size_t component = 0u; component < dimension; ++component) {
    const double residual = fixture.vector.fields[0][field_begin + component] -
                            fixture.state.current_inputs[begin + component];
    fixture.scratch.residual[component] = residual;
    square += residual * residual;
    maximum = std::max(maximum, std::abs(residual));
  }
  const double rms = std::sqrt(square) / std::sqrt(static_cast<double>(dimension));
  return {fixture.scratch.residual,
          dimension,
          rms,
          maximum,
          damping,
          restart_history,
          fixture.unit_metric.data(),
          std::numeric_limits<double>::max()};
}

int test_qsh_only_ragged_mix_restart_and_transaction() {
  std::string error;
  Fixture fixture(error);
  CHECK(fixture.plan.sealed());
  CHECK(fixture.plan.vector_offsets() == std::vector<std::int64_t>({0, 2, 6}));
  CHECK(fixture.plan.total_vector_elements() == 6);
  CHECK(fixture.plan.maximum_vector_elements() == 4);
  CHECK(fixture.plan.matches_vector_layout(fixture.layout));

  /* Guard bytes prove a qsh-only plan never assumes adjacent D/Q fields. */
  std::memset(fixture.vector_storage.data(), 0x5a, Fixture::kQshOffset);
  std::memset(static_cast<std::byte*>(fixture.vector_storage.data()) + Fixture::kQshEndOffset, 0x5a,
              Fixture::kSuffixBytes);
  const double initial[6]{0.10, -0.20, 0.30, -0.40, 0.05, -0.06};
  std::copy(std::begin(initial), std::end(initial), fixture.vector.fields[0]);
  CHECK(xtbloom::detail::common::initialize_scc_mixer_state_cpu(
            fixture.plan, fixture.vector, fixture.state, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(equal(initial, fixture.state.current_inputs, 6u));
  CHECK(fixture.state.initialized[0] == 1u && fixture.state.initialized[1] == 1u);

  const std::vector<std::byte> prefix_before(
      static_cast<const std::byte*>(fixture.vector_storage.data()),
      static_cast<const std::byte*>(fixture.vector_storage.data()) + Fixture::kQshOffset);
  const std::vector<std::byte> suffix_before(
      static_cast<const std::byte*>(fixture.vector_storage.data()) + Fixture::kQshEndOffset,
      static_cast<const std::byte*>(fixture.vector_storage.data()) + Fixture::kVectorBytes);

  fixture.vector.fields[0][0] = initial[0] + 0.02;
  fixture.vector.fields[0][1] = initial[1] - 0.01;
  CHECK(xtbloom::detail::common::mix_scc_broyden_system_cpu(fixture.plan, 0, fixture.vector,
                                                            fixture.state, fixture.scratch,
                                                            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::abs(fixture.vector.fields[0][0] - (initial[0] + 0.4 * 0.02)) < 1.0e-15);
  CHECK(std::abs(fixture.vector.fields[0][1] - (initial[1] - 0.4 * 0.01)) < 1.0e-15);
  CHECK(fixture.state.iterations[0] == 1u && fixture.state.iterations[1] == 0u);
  CHECK(std::equal(prefix_before.begin(), prefix_before.end(),
                   static_cast<const std::byte*>(fixture.vector_storage.data())));
  CHECK(std::equal(
      suffix_before.begin(), suffix_before.end(),
      static_cast<const std::byte*>(fixture.vector_storage.data()) + Fixture::kQshEndOffset));

  AlignedBuffer staged_storage(fixture.plan.state_size_bytes());
  SccMixerState staged;
  CHECK(xtbloom::detail::common::bind_scc_mixer_state(fixture.plan, staged_storage.data(),
                                                      staged_storage.size(), staged,
                                                      error) == XTBLOOM_STATUS_SUCCESS);
  const std::vector<std::byte> state_before(
      static_cast<const std::byte*>(fixture.state.workspace_base),
      static_cast<const std::byte*>(fixture.state.workspace_base) +
          fixture.state.workspace_size_bytes);
  CHECK(xtbloom::detail::common::commit_scc_mixer_system_transaction_cpu(
            fixture.plan, 1, staged, fixture.state, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::equal(state_before.begin(), state_before.end(),
                   static_cast<const std::byte*>(fixture.state.workspace_base)));
  std::memset(staged_storage.data(), 0x3c, staged_storage.size());
  CHECK(xtbloom::detail::common::prepare_scc_mixer_system_transaction_cpu(
            fixture.plan, 1, fixture.state, staged, error) == XTBLOOM_STATUS_SUCCESS);
  const double first_system_sentinel = staged.current_inputs[0];
  CHECK(xtbloom::detail::common::commit_scc_mixer_system_transaction_cpu(
            fixture.plan, 1, staged, fixture.state, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.state.current_inputs[0] != first_system_sentinel);

  const double restarted[4]{-0.1, 0.2, -0.3, 0.4};
  std::copy(std::begin(restarted), std::end(restarted), fixture.vector.fields[0] + 2);
  CHECK(xtbloom::detail::common::restart_scc_mixer_system_cpu(
            fixture.plan, 1, fixture.vector, fixture.state, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.state.restart_counts[1] == 1u);
  CHECK(fixture.state.iterations[1] == 0u);
  CHECK(equal(restarted, fixture.state.current_inputs + 2, 4u));
  CHECK(fixture.state.iterations[0] == 1u);

  const double current_before[4]{fixture.state.current_inputs[2], fixture.state.current_inputs[3],
                                 fixture.state.current_inputs[4], fixture.state.current_inputs[5]};
  fixture.vector.fields[0][2] = std::numeric_limits<double>::quiet_NaN();
  CHECK(xtbloom::detail::common::mix_scc_broyden_system_cpu(
            fixture.plan, 1, fixture.vector, fixture.state, fixture.scratch, error) ==
        XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(equal(current_before, fixture.state.current_inputs + 2, 4u));
  CHECK(fixture.state.iterations[1] == 0u);
  CHECK(fixture.state.system_statuses[1] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(fixture.state.system_statuses[0] == XTBLOOM_STATUS_SUCCESS);
  return 0;
}

int test_layout_rejects_fake_or_overlapping_fields() {
  std::string error;
  std::int64_t offsets[2]{0, 2};
  SccMixerVectorLayoutView layout;
  layout.batch_size = 1;
  layout.workspace_size_bytes = 64u;
  layout.workspace_alignment = 64u;
  layout.field_count = 2u;
  layout.fields[0] = {0u, 2u * sizeof(double), 2, offsets, 2u};
  layout.fields[1] = layout.fields[0];
  SccMixerPlan plan;
  CHECK(xtbloom::detail::common::make_scc_mixer_plan(layout, 3, 0.4, 1.0e-8, 1.0e-8, plan, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(!plan.sealed());
  layout.field_count = 1u;
  offsets[1] = 0;
  CHECK(xtbloom::detail::common::make_scc_mixer_plan(layout, 3, 0.4, 1.0e-8, 1.0e-8, plan, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  return 0;
}

int test_prepared_identity_path_matches_baseline_bitwise() {
  std::string first_error;
  std::string second_error;
  Fixture baseline(first_error);
  Fixture prepared(second_error);
  CHECK(baseline.plan.sealed() && prepared.plan.sealed());
  const double initial[6]{0.10, -0.20, 0.30, -0.40, 0.05, -0.06};
  std::copy(std::begin(initial), std::end(initial), baseline.vector.fields[0]);
  std::copy(std::begin(initial), std::end(initial), prepared.vector.fields[0]);
  CHECK(xtbloom::detail::common::initialize_scc_mixer_state_cpu(
            baseline.plan, baseline.vector, baseline.state, first_error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::common::initialize_scc_mixer_state_cpu(prepared.plan, prepared.vector,
                                                                prepared.state, second_error) ==
        XTBLOOM_STATUS_SUCCESS);

  for (int iteration = 0; iteration < 2; ++iteration) {
    baseline.vector.fields[0][0] += 0.02 + 0.01 * iteration;
    baseline.vector.fields[0][1] -= 0.01 + 0.005 * iteration;
    prepared.vector.fields[0][0] = baseline.vector.fields[0][0];
    prepared.vector.fields[0][1] = baseline.vector.fields[0][1];
    const SccMixerPreparedStepView view = prepare_raw_residual(prepared, 0u, 0.4);
    CHECK(xtbloom::detail::common::mix_scc_broyden_system_cpu(
              baseline.plan, 0, baseline.vector, baseline.state, baseline.scratch, first_error) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom::detail::common::mix_scc_broyden_system_cpu_prepared(
              prepared.plan, 0, prepared.vector, prepared.state, prepared.scratch, view,
              second_error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(equal(baseline.state.current_inputs, prepared.state.current_inputs, 6u));
    CHECK(equal(baseline.state.previous_inputs, prepared.state.previous_inputs, 6u));
    CHECK(equal(baseline.state.previous_residuals, prepared.state.previous_residuals, 6u));
    CHECK(equal(baseline.state.df_history, prepared.state.df_history, 18u));
    CHECK(equal(baseline.state.u_history, prepared.state.u_history, 18u));
    CHECK(equal(baseline.state.omega, prepared.state.omega, 6u));
    CHECK(equal(baseline.state.residual_rms, prepared.state.residual_rms, 2u));
    CHECK(equal(baseline.state.residual_maximum, prepared.state.residual_maximum, 2u));
    CHECK(std::equal(baseline.state.iterations, baseline.state.iterations + 2u,
                     prepared.state.iterations));
    CHECK(std::equal(baseline.state.history_ages, baseline.state.history_ages + 2u,
                     prepared.state.history_ages));
    CHECK(std::memcmp(baseline.state.workspace_base, prepared.state.workspace_base,
                      baseline.plan.state_size_bytes()) == 0);
    CHECK(std::memcmp(baseline.vector.workspace_base, prepared.vector.workspace_base,
                      Fixture::kVectorBytes) == 0);
  }
  return 0;
}

int test_prepared_damping_change_restarts_only_secant_history() {
  std::string error;
  Fixture fixture(error);
  CHECK(fixture.plan.sealed());
  const double initial[6]{0.10, -0.20, 0.30, -0.40, 0.05, -0.06};
  std::copy(std::begin(initial), std::end(initial), fixture.vector.fields[0]);
  CHECK(xtbloom::detail::common::initialize_scc_mixer_state_cpu(
            fixture.plan, fixture.vector, fixture.state, error) == XTBLOOM_STATUS_SUCCESS);
  for (int iteration = 0; iteration < 2; ++iteration) {
    fixture.vector.fields[0][0] += 0.02;
    fixture.vector.fields[0][1] -= 0.01;
    CHECK(xtbloom::detail::common::mix_scc_broyden_system_cpu(fixture.plan, 0, fixture.vector,
                                                              fixture.state, fixture.scratch,
                                                              error) == XTBLOOM_STATUS_SUCCESS);
  }
  CHECK(fixture.state.history_ages[0] == 2u);
  const std::uint64_t iterations_before = fixture.state.iterations[0];
  const std::uint64_t explicit_restarts_before = fixture.state.restart_counts[0];
  const double current_before[2]{fixture.state.current_inputs[0], fixture.state.current_inputs[1]};
  fixture.vector.fields[0][0] += 0.04;
  fixture.vector.fields[0][1] -= 0.02;
  SccMixerPreparedStepView view = prepare_raw_residual(fixture, 0u, 0.2, true);
  const double residual[2]{view.effective_residual[0], view.effective_residual[1]};
  view.raw_residual_rms = 7.0;
  view.raw_residual_maximum = 8.0;
  CHECK(xtbloom::detail::common::mix_scc_broyden_system_cpu_prepared(
            fixture.plan, 0, fixture.vector, fixture.state, fixture.scratch, view, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.state.iterations[0] == iterations_before + 1u);
  CHECK(fixture.state.history_ages[0] == 1u);
  CHECK(fixture.state.restart_counts[0] == explicit_restarts_before);
  CHECK(fixture.state.residual_rms[0] == 7.0);
  CHECK(fixture.state.residual_maximum[0] == 8.0);
  CHECK(fixture.state.converged[0] == 0u);
  CHECK(fixture.vector.fields[0][0] == current_before[0] + 0.2 * residual[0]);
  CHECK(fixture.vector.fields[0][1] == current_before[1] + 0.2 * residual[1]);
  CHECK(std::all_of(fixture.state.df_history, fixture.state.df_history + 6u,
                    [](double value) { return value == 0.0; }));
  CHECK(std::all_of(fixture.state.u_history, fixture.state.u_history + 6u,
                    [](double value) { return value == 0.0; }));
  CHECK(std::all_of(fixture.state.omega, fixture.state.omega + 3u,
                    [](double value) { return value == 0.0; }));
  return 0;
}

int test_prepared_trust_radius_caps_complete_broyden_step() {
  std::string error;
  Fixture fixture(error);
  CHECK(fixture.plan.sealed());
  std::fill_n(fixture.vector.fields[0], Fixture::kQshElements, 0.0);
  CHECK(xtbloom::detail::common::initialize_scc_mixer_state_cpu(
            fixture.plan, fixture.vector, fixture.state, error) == XTBLOOM_STATUS_SUCCESS);

  /* Construct a finite but very large Broyden correction through the previous
   * input displacement. The cap must apply to the complete candidate, not only
   * to damping * residual. */
  fixture.state.iterations[0] = 1u;
  fixture.state.history_ages[0] = 1u;
  fixture.state.previous_inputs[0] = -1000.0;
  fixture.state.previous_residuals[0] = 0.0;
  fixture.vector.fields[0][0] = 1.0;
  SccMixerPreparedStepView view = prepare_raw_residual(fixture, 0u, 0.4);
  view.maximum_weighted_step_norm = 0.5;
  CHECK(xtbloom::detail::common::mix_scc_broyden_system_cpu_prepared(
            fixture.plan, 0, fixture.vector, fixture.state, fixture.scratch, view, error) ==
        XTBLOOM_STATUS_SUCCESS);
  const double step0 = fixture.state.current_inputs[0];
  const double step1 = fixture.state.current_inputs[1];
  CHECK(std::abs(std::hypot(step0, step1) - 0.5) < 1.0e-14);
  CHECK(step0 < 0.0);
  CHECK(fixture.state.history_ages[0] == 2u);
  CHECK(fixture.state.iterations[0] == 2u);
  return 0;
}

int test_prepared_trust_radius_handles_finite_endpoints_with_overflowing_double_difference() {
  std::string error;
  Fixture fixture(error);
  CHECK(fixture.plan.sealed());
  std::fill_n(fixture.vector.fields[0], Fixture::kQshElements, 0.0);
  CHECK(xtbloom::detail::common::initialize_scc_mixer_state_cpu(
            fixture.plan, fixture.vector, fixture.state, error) == XTBLOOM_STATUS_SUCCESS);

  /* Two orthogonal secant corrections move a finite +0.75*DBL_MAX current
   * input to a finite negative candidate. Their difference exceeds DBL_MAX,
   * so the trust-radius calculation must not form it in binary64 or classify
   * it as a structural INVALID_ARGUMENT. */
  const double maximum = std::numeric_limits<double>::max();
  const double current = 0.75 * maximum;
  const double radius = 0.125 * maximum;
  fixture.state.current_inputs[0] = current;
  fixture.state.previous_inputs[0] = 0.0;
  fixture.state.previous_residuals[0] = 1.0;
  fixture.state.previous_residuals[1] = 0.0;
  fixture.state.df_history[0] = 1.0;
  fixture.state.df_history[1] = 0.0;
  fixture.state.u_history[0] = current;
  fixture.state.u_history[1] = 0.0;
  fixture.state.omega[0] = 1.0;
  fixture.state.iterations[0] = 2u;
  fixture.state.history_ages[0] = 2u;
  fixture.scratch.residual[0] = 1.0;
  fixture.scratch.residual[1] = 1.0;

  const std::vector<double> peer_inputs(fixture.state.current_inputs + 2,
                                        fixture.state.current_inputs + 6);
  const std::vector<double> peer_wavefunction(fixture.vector.fields[0] + 2,
                                              fixture.vector.fields[0] + 6);
  const SccMixerPreparedStepView view{fixture.scratch.residual,   2u,    1.0, 1.0, 0.4, false,
                                      fixture.unit_metric.data(), radius};
  CHECK(xtbloom::detail::common::mix_scc_broyden_system_cpu_prepared(
            fixture.plan, 0, fixture.vector, fixture.state, fixture.scratch, view, error) ==
        XTBLOOM_STATUS_SUCCESS);
  const double bounded_step = current - fixture.state.current_inputs[0];
  CHECK(std::isfinite(fixture.state.current_inputs[0]));
  CHECK(std::abs(bounded_step / radius - 1.0) < 1.0e-12);
  CHECK(std::equal(peer_inputs.begin(), peer_inputs.end(), fixture.state.current_inputs + 2));
  CHECK(
      std::equal(peer_wavefunction.begin(), peer_wavefunction.end(), fixture.vector.fields[0] + 2));
  CHECK(fixture.state.iterations[0] == 3u);
  CHECK(fixture.state.history_ages[0] == 3u);
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_qsh_only_ragged_mix_restart_and_transaction(); status != 0) {
    return status;
  }
  if (const int status = test_layout_rejects_fake_or_overlapping_fields(); status != 0) {
    return status;
  }
  if (const int status = test_prepared_identity_path_matches_baseline_bitwise(); status != 0) {
    return status;
  }
  if (const int status = test_prepared_damping_change_restarts_only_secant_history(); status != 0) {
    return status;
  }
  if (const int status = test_prepared_trust_radius_caps_complete_broyden_step(); status != 0) {
    return status;
  }
  return test_prepared_trust_radius_handles_finite_endpoints_with_overflowing_double_difference();
}
