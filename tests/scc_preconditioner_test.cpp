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
};

bool make_fixture(Fixture& fixture, std::string& error, double molecular_charge = 0.0,
                  std::int32_t unpaired_electrons = 0, std::int32_t spin_channel_count = 1) {
  const std::int64_t atom_offsets[2]{0, 2};
  const std::int32_t atomic_numbers[2]{6, 8};
  const double charges[1]{molecular_charge};
  const std::int32_t unpaired[1]{unpaired_electrons};
  const std::int32_t spin_channels[1]{spin_channel_count};
  if (xtbloom::detail::gfn2::make_basis_plan(1, 2, atom_offsets, atomic_numbers, fixture.basis,
                                             error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_wavefunction_layout(fixture.basis, atomic_numbers, charges,
                                                      unpaired, spin_channels, fixture.layout,
                                                      error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_es2_plan(fixture.basis, atomic_numbers, fixture.es2, error) !=
          XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_aes2_plan(fixture.basis, atomic_numbers, fixture.aes2, error) !=
          XTBLOOM_STATUS_SUCCESS ||
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
  return xtbloom::detail::gfn2::bind_wavefunction_view(
             fixture.layout, fixture.wavefunction_storage->data(),
             fixture.wavefunction_storage->size(), fixture.wavefunction,
             error) == XTBLOOM_STATUS_SUCCESS &&
         xtbloom::detail::gfn2::bind_scc_mixer_state(fixture.mixer, fixture.state_storage->data(),
                                                     fixture.state_storage->size(), fixture.state,
                                                     error) == XTBLOOM_STATUS_SUCCESS &&
         xtbloom::detail::gfn2::bind_scc_mixer_workspace(
             fixture.mixer, fixture.scratch_storage->data(), fixture.scratch_storage->size(),
             fixture.scratch, error) == XTBLOOM_STATUS_SUCCESS &&
         xtbloom::detail::gfn2::initialize_scc_mixer_state_cpu(
             fixture.mixer, fixture.wavefunction, fixture.state, error) == XTBLOOM_STATUS_SUCCESS;
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
            fixture.wavefunction, fixture.state, fixture.scratch, diagnostics,
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
            fixture.wavefunction, fixture.state, fixture.scratch, diagnostics,
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
            fixture.wavefunction, fixture.state, fixture.scratch, diagnostics,
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
            fixture.wavefunction, fixture.state, fixture.scratch, diagnostics,
            error) == XTBLOOM_STATUS_SUCCESS);
  const xtbloom::detail::gfn2::SccMixerPreparedStepView prepared{
      fixture.scratch.residual,         dimension, diagnostics.raw_residual_rms,
      diagnostics.raw_residual_maximum, 0.4,       false,
      fixture.preconditioner.metric_weights().data(), std::numeric_limits<double>::max()};
  CHECK(xtbloom::detail::gfn2::mix_scc_broyden_system_cpu_prepared(
            fixture.mixer, 0, fixture.wavefunction, fixture.state, fixture.scratch, prepared,
            error) == XTBLOOM_STATUS_SUCCESS);
  install_residual(fixture, residual);
  CHECK(xtbloom::detail::gfn2::prepare_scc_residual_system_cpu(
            fixture.preconditioner, xtbloom::detail::gfn2::SccResidualPolicy::kControllerOnly, 0,
            fixture.wavefunction, fixture.state, fixture.scratch, diagnostics,
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
            fixture.wavefunction, fixture.state, fixture.scratch, diagnostics,
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
            fixture.wavefunction, fixture.state, fixture.scratch, original,
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
            fixture.wavefunction, fixture.state, fixture.scratch, rotated,
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
  return test_weighted_norm_and_angle_are_rotation_invariant();
}
