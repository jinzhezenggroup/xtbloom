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

/* Frozen susceptibility scale for the Phase-2 QEq experiment. It deliberately
 * remains an internal constant until finite-difference response evidence can
 * justify any public or adaptive policy. */
inline constexpr double kSccPairResponseScale = 0.08;

enum class SccResidualPolicy : std::uint8_t {
  kControllerOnly = 0,
  kLocalV1 = 1,
  kPairResponseV1 = 2,
};

struct SccResidualDiagnostics {
  double raw_residual_rms = 0.0;
  double raw_residual_maximum = 0.0;
  double weighted_residual_norm = 0.0;
  double previous_weighted_residual_norm = 0.0;
  double weighted_residual_cosine = 0.0;
  bool has_previous_residual = false;
  bool cosine_is_valid = false;
  bool preconditioner_applied = false;
  bool preconditioner_fell_back = false;
};

/* Geometry-dependent factorization of the constrained shell-space response.
 * H factors use the ES2 ragged dense-matrix packing. constraint_solutions
 * stores H^-1 1 per shell and constraint_denominators stores 1^T H^-1 1 per
 * system. enabled is lane-local: a failed conditioning or Cholesky gate makes
 * only that system use the raw residual. */
struct SccPairResponseGeometryCache {
  double* cholesky_factors = nullptr;
  std::int64_t factor_elements = 0;
  double* constraint_solutions = nullptr;
  std::int64_t constraint_elements = 0;
  double* constraint_denominators = nullptr;
  std::int64_t denominator_elements = 0;
  std::uint8_t* enabled = nullptr;
  std::int64_t enabled_elements = 0;
  std::uint64_t geometry_generation = 0;
  const ES2PlanData* plan_identity = nullptr;
};

/* Caller-owned staging for atomic cache publication. */
struct SccPairResponseWorkspace {
  double* factor_scratch = nullptr;
  std::int64_t factor_elements = 0;
  double* constraint_scratch = nullptr;
  std::int64_t constraint_elements = 0;
  double* denominator_scratch = nullptr;
  std::int64_t denominator_elements = 0;
  std::uint8_t* enabled_scratch = nullptr;
  std::int64_t enabled_elements = 0;
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
  [[nodiscard]] std::int64_t total_shells() const noexcept {
    return shell_offsets_.empty() ? 0 : shell_offsets_.back();
  }
  [[nodiscard]] std::int64_t total_matrix_elements() const noexcept {
    return matrix_offsets_.empty() ? 0 : matrix_offsets_.back();
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
  std::vector<std::int64_t> matrix_offsets_;
  std::vector<std::int64_t> shell_to_atom_;
  std::vector<std::int32_t> spin_channels_;
  std::vector<double> metric_weights_;
  std::vector<double> charge_shell_scales_;
  std::vector<double> shell_hardness_;
  const ES2PlanData* es2_plan_identity_ = nullptr;
  std::int64_t es2_matrix_elements_ = 0;

  friend xtbloom_status_t make_scc_preconditioner_plan(const WavefunctionLayout&, const ES2Plan&,
                                                       const AES2Plan&, SccPreconditionerPlan&,
                                                       std::string&);
  friend xtbloom_status_t prepare_scc_residual_system_cpu(
      const SccPreconditionerPlan&, SccResidualPolicy, std::int64_t, const WavefunctionView&,
      const SccPairResponseGeometryCache&, const SccMixerState&, const SccMixerWorkspace&,
      SccResidualDiagnostics&, std::string&);
  friend xtbloom_status_t update_scc_pair_response_geometry_cache_cpu(
      const SccPreconditionerPlan&, const ES2GeometryCache&, bool, const SccPairResponseWorkspace&,
      SccPairResponseGeometryCache&, std::string&);
};

xtbloom_status_t make_scc_preconditioner_plan(const WavefunctionLayout& wavefunction,
                                              const ES2Plan& es2, const AES2Plan& aes2,
                                              SccPreconditionerPlan& plan, std::string& error);

/* Build H = diag(gamma/alpha) + K_pair from the production ES2 cache, where
 * K_pair retains only different-atom shell couplings. A sqrt(epsilon)
 * Gershgorin reciprocal-condition floor, a scale-aware constraint denominator
 * bound, and Cholesky success are required per system. Periodic A response is
 * not yet included; periodic_response_enabled therefore disables the
 * experimental map lane-locally instead of silently solving a different
 * Jacobian. A caller-owned b shift alone does not disable it. Scratch is
 * published atomically into cache on success. */
xtbloom_status_t update_scc_pair_response_geometry_cache_cpu(
    const SccPreconditionerPlan& plan, const ES2GeometryCache& es2_cache,
    bool periodic_response_enabled, const SccPairResponseWorkspace& workspace,
    SccPairResponseGeometryCache& cache, std::string& error);

/*
 * Pack the raw q/d/Q residual into mixer_workspace.residual, optionally apply
 * the local-v1 charge-shell scaling or the pair-response-v1 constrained QEq
 * solve, and compute raw/public plus weighted controller diagnostics.
 *
 * pair-response-v1 solves H y = G r and removes the H^-1 1 constraint mode,
 * with G_ss = gamma_s/alpha and H = G + K_pair. K_pair is exactly the
 * production ES2 shell coupling for shells on different atoms and zero for
 * same-atom shells. Only the first shell-charge channel is transformed;
 * magnetization, dipoles, and quadrupoles remain bitwise raw. This is a
 * QEq-style numerical experiment, not a validated electronic susceptibility.
 * A non-tangent raw residual or numerical/cache failure restores the exact raw
 * residual and marks a deterministic controller fallback. Successful
 * steady-state calls allocate nothing and perform no eigensolve or model-energy
 * evaluation.
 */
xtbloom_status_t prepare_scc_residual_system_cpu(
    const SccPreconditionerPlan& plan, SccResidualPolicy policy, std::int64_t system,
    const WavefunctionView& raw_wavefunction, const SccPairResponseGeometryCache& pair_cache,
    const SccMixerState& mixer_state, const SccMixerWorkspace& mixer_workspace,
    SccResidualDiagnostics& diagnostics, std::string& error);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_SCC_PRECONDITIONER_HPP
