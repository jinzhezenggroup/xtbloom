#ifndef XTBLOOM_MODEL_GFN2_SCC_PRECONDITIONER_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_SCC_PRECONDITIONER_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model/gfn2/aes2.hpp"
#include "model/gfn2/es2.hpp"
#include "model/gfn2/scc_mixer.hpp"
#include "model/gfn2/wavefunction.hpp"

namespace xtbloom::detail::gfn2 {

enum class SccResidualPolicy : std::uint8_t {
  kControllerOnly = 0,
  kLocalV1 = 1,
};

struct SccResidualDiagnostics {
  double raw_residual_rms = 0.0;
  double raw_residual_maximum = 0.0;
  double weighted_residual_norm = 0.0;
  double previous_weighted_residual_norm = 0.0;
  double weighted_residual_cosine = 0.0;
  bool has_previous_residual = false;
  bool cosine_is_valid = false;
};

/*
 * Geometry-independent PAIRS-SCC metric and bounded local preconditioner.
 * The packed metric is frozen with the plan: q has unit weight, dipoles use
 * R_A^-2, and packed quadrupoles use the rotation-invariant Frobenius weights
 * [1,2,1,2,2,1] R_A^-4.
 */
class SccPreconditionerPlan {
 public:
  [[nodiscard]] bool sealed() const noexcept { return sealed_; }
  [[nodiscard]] std::int64_t batch_size() const noexcept { return batch_size_; }
  [[nodiscard]] std::int64_t total_vector_elements() const noexcept {
    return vector_offsets_.empty() ? 0 : vector_offsets_.back();
  }
  [[nodiscard]] const std::vector<std::int64_t>& vector_offsets() const noexcept {
    return vector_offsets_;
  }
  [[nodiscard]] const std::vector<double>& metric_weights() const noexcept {
    return metric_weights_;
  }
  [[nodiscard]] const std::vector<double>& charge_shell_scales() const noexcept {
    return charge_shell_scales_;
  }
  [[nodiscard]] std::size_t resident_bytes() const noexcept;
  [[nodiscard]] bool overlaps_storage(const void* data, std::size_t size_bytes) const noexcept;

 private:
  bool sealed_ = false;
  std::int64_t batch_size_ = 0;
  std::vector<std::int64_t> vector_offsets_;
  std::vector<std::int64_t> q_system_offsets_;
  std::vector<std::int64_t> dipole_system_offsets_;
  std::vector<std::int64_t> quadrupole_system_offsets_;
  std::vector<std::int64_t> atom_offsets_;
  std::vector<std::int64_t> shell_offsets_;
  std::vector<std::int32_t> spin_channels_;
  std::vector<double> metric_weights_;
  std::vector<double> charge_shell_scales_;

  friend xtbloom_status_t make_scc_preconditioner_plan(const WavefunctionLayout&, const ES2Plan&,
                                                       const AES2Plan&, SccPreconditionerPlan&,
                                                       std::string&);
  friend xtbloom_status_t prepare_scc_residual_system_cpu(
      const SccPreconditionerPlan&, SccResidualPolicy, std::int64_t, const WavefunctionView&,
      const SccMixerState&, const SccMixerWorkspace&, SccResidualDiagnostics&, std::string&);
};

xtbloom_status_t make_scc_preconditioner_plan(const WavefunctionLayout& wavefunction,
                                              const ES2Plan& es2, const AES2Plan& aes2,
                                              SccPreconditionerPlan& plan, std::string& error);

/*
 * Pack the raw q/d/Q residual into mixer_workspace.residual, optionally apply
 * the local-v1 charge-shell scaling, and compute raw/public plus weighted
 * controller diagnostics. Successful steady-state calls allocate nothing.
 */
xtbloom_status_t prepare_scc_residual_system_cpu(const SccPreconditionerPlan& plan,
                                                 SccResidualPolicy policy, std::int64_t system,
                                                 const WavefunctionView& raw_wavefunction,
                                                 const SccMixerState& mixer_state,
                                                 const SccMixerWorkspace& mixer_workspace,
                                                 SccResidualDiagnostics& diagnostics,
                                                 std::string& error);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_SCC_PRECONDITIONER_HPP
