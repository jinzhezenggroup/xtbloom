#include "model/gfn2/scc_preconditioner.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

namespace xtbloom::detail::gfn2 {
namespace {

constexpr double kMinimumChargeShellScale = 0.5;
constexpr double kMaximumChargeShellScale = 2.0;

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool make_range(const void* pointer, std::size_t bytes, AddressRange& range) noexcept {
  if (pointer == nullptr || bytes == 0u) {
    return false;
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  range = {begin, begin + bytes};
  return true;
}

bool ranges_overlap(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin < second.end && second.begin < first.end;
}

template <typename T>
bool vector_overlaps(const std::vector<T>& values, const AddressRange& range) noexcept {
  if (values.empty()) {
    return false;
  }
  AddressRange candidate;
  return !make_range(values.data(), values.capacity() * sizeof(T), candidate) ||
         ranges_overlap(candidate, range);
}

bool finite_positive(double value) noexcept { return std::isfinite(value) && value > 0.0; }

}  // namespace

std::size_t SccPreconditionerPlan::resident_bytes() const noexcept {
  return sizeof(*this) + vector_offsets_.capacity() * sizeof(std::int64_t) +
         q_system_offsets_.capacity() * sizeof(std::int64_t) +
         dipole_system_offsets_.capacity() * sizeof(std::int64_t) +
         quadrupole_system_offsets_.capacity() * sizeof(std::int64_t) +
         atom_offsets_.capacity() * sizeof(std::int64_t) +
         shell_offsets_.capacity() * sizeof(std::int64_t) +
         spin_channels_.capacity() * sizeof(std::int32_t) +
         metric_weights_.capacity() * sizeof(double) +
         charge_shell_scales_.capacity() * sizeof(double);
}

bool SccPreconditionerPlan::overlaps_storage(const void* data,
                                             std::size_t size_bytes) const noexcept {
  AddressRange range;
  if (!sealed_ || size_bytes == 0u || !make_range(data, size_bytes, range)) {
    return size_bytes != 0u;
  }
  AddressRange object;
  if (!make_range(this, sizeof(*this), object) || ranges_overlap(object, range)) {
    return true;
  }
  return vector_overlaps(vector_offsets_, range) || vector_overlaps(q_system_offsets_, range) ||
         vector_overlaps(dipole_system_offsets_, range) ||
         vector_overlaps(quadrupole_system_offsets_, range) ||
         vector_overlaps(atom_offsets_, range) || vector_overlaps(shell_offsets_, range) ||
         vector_overlaps(spin_channels_, range) || vector_overlaps(metric_weights_, range) ||
         vector_overlaps(charge_shell_scales_, range);
}

xtbloom_status_t make_scc_preconditioner_plan(const WavefunctionLayout& wavefunction,
                                              const ES2Plan& es2, const AES2Plan& aes2,
                                              SccPreconditionerPlan& plan, std::string& error) {
  if (wavefunction.batch_size <= 0 || !es2.sealed() || !aes2.sealed() ||
      es2.batch_size() != wavefunction.batch_size || aes2.batch_size() != wavefunction.batch_size ||
      es2.total_atoms() != wavefunction.total_atoms ||
      aes2.total_atoms() != wavefunction.total_atoms ||
      es2.total_shells() != wavefunction.total_shells ||
      es2.atom_offsets() != wavefunction.atom_offsets ||
      aes2.atom_offsets() != wavefunction.atom_offsets ||
      es2.batch_shell_offsets() != wavefunction.batch_shell_offsets ||
      wavefunction.spin_channels.size() != static_cast<std::size_t>(wavefunction.batch_size) ||
      es2.shell_hardness().size() != static_cast<std::size_t>(wavefunction.total_shells) ||
      aes2.multipole_radius().size() != static_cast<std::size_t>(wavefunction.total_atoms)) {
    error = "PAIRS-SCC preconditioner inputs do not describe one compatible GFN2 topology";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  try {
    SccPreconditionerPlan created;
    created.batch_size_ = wavefunction.batch_size;
    created.q_system_offsets_ = wavefunction.qsh.system_offsets;
    created.dipole_system_offsets_ = wavefunction.dipole.system_offsets;
    created.quadrupole_system_offsets_ = wavefunction.quadrupole.system_offsets;
    created.atom_offsets_ = wavefunction.atom_offsets;
    created.shell_offsets_ = wavefunction.batch_shell_offsets;
    created.spin_channels_ = wavefunction.spin_channels;
    created.charge_shell_scales_.assign(static_cast<std::size_t>(wavefunction.qsh.element_count),
                                        1.0);
    created.vector_offsets_.assign(static_cast<std::size_t>(wavefunction.batch_size) + 1u, 0);

    for (std::size_t system = 0u; system < static_cast<std::size_t>(wavefunction.batch_size);
         ++system) {
      const std::int64_t atom_count =
          wavefunction.atom_offsets[system + 1u] - wavefunction.atom_offsets[system];
      const std::int64_t shell_count =
          wavefunction.batch_shell_offsets[system + 1u] - wavefunction.batch_shell_offsets[system];
      const std::int64_t nspin = wavefunction.spin_channels[system];
      const std::int64_t q_count =
          wavefunction.qsh.system_offsets[system + 1u] - wavefunction.qsh.system_offsets[system];
      const std::int64_t d_count = wavefunction.dipole.system_offsets[system + 1u] -
                                   wavefunction.dipole.system_offsets[system];
      const std::int64_t q2_count = wavefunction.quadrupole.system_offsets[system + 1u] -
                                    wavefunction.quadrupole.system_offsets[system];
      if ((nspin != 1 && nspin != 2) || atom_count <= 0 || shell_count <= 0 ||
          q_count != nspin * shell_count || d_count != 3 * nspin * atom_count ||
          q2_count != 6 * nspin * atom_count) {
        error = "PAIRS-SCC preconditioner encountered an invalid q/d/Q field layout";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      const std::int64_t dimension = q_count + d_count + q2_count;
      if (created.vector_offsets_[system] > std::numeric_limits<std::int64_t>::max() - dimension) {
        error = "PAIRS-SCC preconditioner vector offsets overflow int64_t";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      created.vector_offsets_[system + 1u] = created.vector_offsets_[system] + dimension;

      double inverse_hardness_sum = 0.0;
      const std::size_t shell_begin =
          static_cast<std::size_t>(wavefunction.batch_shell_offsets[system]);
      const std::size_t shell_end =
          static_cast<std::size_t>(wavefunction.batch_shell_offsets[system + 1u]);
      for (std::size_t shell = shell_begin; shell < shell_end; ++shell) {
        const double hardness = es2.shell_hardness()[shell];
        if (!finite_positive(hardness)) {
          error = "PAIRS-SCC charge-shell hardness must be finite and positive";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        inverse_hardness_sum += 1.0 / hardness;
      }
      const double reference_hardness = static_cast<double>(shell_count) / inverse_hardness_sum;
      if (!finite_positive(reference_hardness)) {
        error = "PAIRS-SCC harmonic charge-shell hardness is not finite and positive";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      const std::size_t q_begin = static_cast<std::size_t>(wavefunction.qsh.system_offsets[system]);
      for (std::int64_t channel = 0; channel < nspin; ++channel) {
        for (std::size_t local_shell = 0u; local_shell < static_cast<std::size_t>(shell_count);
             ++local_shell) {
          const double hardness = es2.shell_hardness()[shell_begin + local_shell];
          created.charge_shell_scales_[q_begin + static_cast<std::size_t>(channel * shell_count) +
                                       local_shell] =
              std::clamp(reference_hardness / hardness, kMinimumChargeShellScale,
                         kMaximumChargeShellScale);
        }
      }
    }

    created.metric_weights_.reserve(static_cast<std::size_t>(created.vector_offsets_.back()));
    constexpr std::array<double, 6u> quadrupole_frobenius{{1.0, 2.0, 1.0, 2.0, 2.0, 1.0}};
    for (std::size_t system = 0u; system < static_cast<std::size_t>(wavefunction.batch_size);
         ++system) {
      const std::size_t atom_begin = static_cast<std::size_t>(wavefunction.atom_offsets[system]);
      const std::size_t atom_end = static_cast<std::size_t>(wavefunction.atom_offsets[system + 1u]);
      const std::size_t atom_count = atom_end - atom_begin;
      const std::size_t shell_count = static_cast<std::size_t>(
          wavefunction.batch_shell_offsets[system + 1u] - wavefunction.batch_shell_offsets[system]);
      const std::size_t nspin = static_cast<std::size_t>(wavefunction.spin_channels[system]);
      created.metric_weights_.insert(created.metric_weights_.end(), nspin * shell_count, 1.0);
      for (std::size_t channel = 0u; channel < nspin; ++channel) {
        for (std::size_t local_atom = 0u; local_atom < atom_count; ++local_atom) {
          const double radius = aes2.multipole_radius()[atom_begin + local_atom];
          if (!finite_positive(radius)) {
            error = "PAIRS-SCC multipole radius must be finite and positive";
            return XTBLOOM_STATUS_INVALID_ARGUMENT;
          }
          const double inverse_radius2 = 1.0 / (radius * radius);
          created.metric_weights_.insert(created.metric_weights_.end(), 3u, inverse_radius2);
        }
      }
      for (std::size_t channel = 0u; channel < nspin; ++channel) {
        for (std::size_t local_atom = 0u; local_atom < atom_count; ++local_atom) {
          const double radius = aes2.multipole_radius()[atom_begin + local_atom];
          const double inverse_radius2 = 1.0 / (radius * radius);
          const double inverse_radius4 = inverse_radius2 * inverse_radius2;
          for (double component_weight : quadrupole_frobenius) {
            created.metric_weights_.push_back(component_weight * inverse_radius4);
          }
        }
      }
    }
    if (created.metric_weights_.size() !=
        static_cast<std::size_t>(created.vector_offsets_.back())) {
      error = "PAIRS-SCC metric packing does not match the q/d/Q vector";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    created.sealed_ = true;
    plan = std::move(created);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate PAIRS-SCC preconditioner metadata";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

xtbloom_status_t prepare_scc_residual_system_cpu(const SccPreconditionerPlan& plan,
                                                 SccResidualPolicy policy, std::int64_t system,
                                                 const WavefunctionView& raw_wavefunction,
                                                 const SccMixerState& mixer_state,
                                                 const SccMixerWorkspace& mixer_workspace,
                                                 SccResidualDiagnostics& diagnostics,
                                                 std::string& error) {
  if (!plan.sealed_ || system < 0 || system >= plan.batch_size_ ||
      raw_wavefunction.qsh == nullptr || raw_wavefunction.dipole == nullptr ||
      raw_wavefunction.quadrupole == nullptr || mixer_state.current_inputs == nullptr ||
      mixer_state.previous_residuals == nullptr || mixer_state.history_ages == nullptr ||
      mixer_workspace.residual == nullptr) {
    error = "PAIRS-SCC residual preparation received an invalid binding";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t index = static_cast<std::size_t>(system);
  const std::size_t vector_begin = static_cast<std::size_t>(plan.vector_offsets_[index]);
  const std::size_t vector_end = static_cast<std::size_t>(plan.vector_offsets_[index + 1u]);
  const std::size_t dimension = vector_end - vector_begin;
  const double* const current = mixer_state.current_inputs + vector_begin;

  double raw_square = 0.0;
  double raw_maximum = 0.0;
  bool finite = true;
  std::size_t packed = 0u;
  const std::array<const double*, 3u> fields{
      {raw_wavefunction.qsh, raw_wavefunction.dipole, raw_wavefunction.quadrupole}};
  const std::array<const std::vector<std::int64_t>*, 3u> offsets{
      {&plan.q_system_offsets_, &plan.dipole_system_offsets_, &plan.quadrupole_system_offsets_}};
  for (std::size_t field = 0u; field < fields.size(); ++field) {
    const std::size_t begin = static_cast<std::size_t>((*offsets[field])[index]);
    const std::size_t end = static_cast<std::size_t>((*offsets[field])[index + 1u]);
    for (std::size_t source = begin; source < end; ++source, ++packed) {
      const double residual = fields[field][source] - current[packed];
      mixer_workspace.residual[packed] = residual;
      finite = finite && std::isfinite(fields[field][source]) && std::isfinite(current[packed]) &&
               std::isfinite(residual);
      if (finite) {
        raw_square += residual * residual;
        raw_maximum = std::max(raw_maximum, std::abs(residual));
        finite = std::isfinite(raw_square);
      }
    }
  }
  if (!finite || packed != dimension) {
    error = "PAIRS-SCC raw residual contains or produces NaN or infinity";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  if (policy == SccResidualPolicy::kLocalV1) {
    const std::size_t shell_count =
        static_cast<std::size_t>(plan.shell_offsets_[index + 1u] - plan.shell_offsets_[index]);
    const std::size_t nspin = static_cast<std::size_t>(plan.spin_channels_[index]);
    const std::size_t q_field_begin = static_cast<std::size_t>(plan.q_system_offsets_[index]);
    for (std::size_t channel = 0u; channel < nspin; ++channel) {
      double raw_sum = 0.0;
      double projected_sum = 0.0;
      double scale_sum = 0.0;
      for (std::size_t shell = 0u; shell < shell_count; ++shell) {
        const std::size_t packed_shell = channel * shell_count + shell;
        const double scale =
            channel == 0u ? plan.charge_shell_scales_[q_field_begin + packed_shell] : 1.0;
        raw_sum += mixer_workspace.residual[packed_shell];
        mixer_workspace.residual[packed_shell] *= scale;
        projected_sum += mixer_workspace.residual[packed_shell];
        scale_sum += scale;
      }
      if (!std::isfinite(raw_sum) || !std::isfinite(projected_sum) || !finite_positive(scale_sum)) {
        error = "PAIRS-SCC charge constraint projection is not finite";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      /* Preserve the raw channel sum rather than forcing it to zero. The
       * initial unrestricted SAD state may not yet carry the target total
       * magnetization, so a zero-sum projection would remove the only update
       * that can enter the constraint manifold. Once there, raw_sum is zero
       * and this reduces to the usual tangent-space projection. */
      const double correction = (projected_sum - raw_sum) / scale_sum;
      for (std::size_t shell = 0u; shell < shell_count; ++shell) {
        const std::size_t packed_shell = channel * shell_count + shell;
        const double scale =
            channel == 0u ? plan.charge_shell_scales_[q_field_begin + packed_shell] : 1.0;
        mixer_workspace.residual[packed_shell] -= scale * correction;
      }
    }
  } else if (policy != SccResidualPolicy::kControllerOnly) {
    error = "PAIRS-SCC residual policy is not supported";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  double weighted_square = 0.0;
  double previous_weighted_square = 0.0;
  double weighted_dot = 0.0;
  const bool has_previous = mixer_state.history_ages[index] > 0u;
  for (std::size_t component = 0u; component < dimension; ++component) {
    const double weight = plan.metric_weights_[vector_begin + component];
    const double residual = mixer_workspace.residual[component];
    weighted_square += weight * residual * residual;
    if (has_previous) {
      const double previous = mixer_state.previous_residuals[vector_begin + component];
      previous_weighted_square += weight * previous * previous;
      weighted_dot += weight * residual * previous;
    }
    if (!std::isfinite(weighted_square) || !std::isfinite(previous_weighted_square) ||
        !std::isfinite(weighted_dot)) {
      error = "PAIRS-SCC weighted residual diagnostic is not finite";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
  }

  diagnostics = {};
  diagnostics.raw_residual_rms = std::sqrt(raw_square) / std::sqrt(static_cast<double>(dimension));
  diagnostics.raw_residual_maximum = raw_maximum;
  diagnostics.weighted_residual_norm = std::sqrt(weighted_square);
  diagnostics.previous_weighted_residual_norm = std::sqrt(previous_weighted_square);
  diagnostics.has_previous_residual = has_previous;
  if (has_previous && diagnostics.weighted_residual_norm > 0.0 &&
      diagnostics.previous_weighted_residual_norm > 0.0) {
    diagnostics.weighted_residual_cosine = std::clamp(
        weighted_dot /
            (diagnostics.weighted_residual_norm * diagnostics.previous_weighted_residual_norm),
        -1.0, 1.0);
    diagnostics.cosine_is_valid = std::isfinite(diagnostics.weighted_residual_cosine);
  }
  if (!std::isfinite(diagnostics.raw_residual_rms) ||
      !std::isfinite(diagnostics.weighted_residual_norm) ||
      !std::isfinite(diagnostics.previous_weighted_residual_norm)) {
    error = "PAIRS-SCC residual norms are not finite";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn2
