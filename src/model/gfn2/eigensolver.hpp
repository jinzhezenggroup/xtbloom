#ifndef XTBLOOM_MODEL_GFN2_EIGENSOLVER_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_EIGENSOLVER_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "model/gfn2/wavefunction.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {

inline constexpr std::size_t kEigensolverWorkspaceAlignment = 64u;
using LapackInt = std::int32_t;

using LapackDpotrfWork = LapackInt (*)(LapackInt matrix_layout, char uplo, LapackInt n,
                                       double* matrix, LapackInt leading_dimension);
using LapackDpoconWork = LapackInt (*)(LapackInt matrix_layout, char uplo, LapackInt n,
                                       const double* factor, LapackInt leading_dimension,
                                       double matrix_one_norm, double* reciprocal_condition,
                                       double* work, LapackInt* integer_work);
using LapackDsyevdWork = LapackInt (*)(LapackInt matrix_layout, char job_vectors, char uplo,
                                       LapackInt n, double* matrix, LapackInt leading_dimension,
                                       double* eigenvalues, double* work, LapackInt work_count,
                                       LapackInt* integer_work, LapackInt integer_work_count);
using CblasDtrsm = void (*)(int layout, int side, int triangle, int transpose, int diagonal,
                            LapackInt rows, LapackInt columns, double alpha,
                            const double* triangular_matrix, LapackInt leading_triangular,
                            double* right_hand_side, LapackInt leading_rhs);
using CblasDgemm = void (*)(int layout, int transpose_left, int transpose_right, LapackInt rows,
                            LapackInt columns, LapackInt inner, double alpha, const double* left,
                            LapackInt leading_left, const double* right, LapackInt leading_right,
                            double beta, double* result, LapackInt leading_result);
using BlasSetNumThreadsLocal = int (*)(int threads);

/*
 * Verified LP64 linear-algebra dispatch.
 *
 * Production code obtains this handle from make_mkl_rt_lp64_backend. The
 * factory loads and verifies all required symbols from a private bundled
 * OpenBLAS provider in native wheels, the configured native LP64 runtime, or
 * common system SONAMEs. System providers must expose local thread control so
 * xtbloom's outer batch workers can keep BLAS sequential. The macOS/Windows
 * wheel provider is a renamed private image instead: initialization fixes that
 * image globally to one thread once, without mutating an unrelated host BLAS.
 *
 * The MKL path is host-isolated. CMake builds a private shim with fixed
 * DT_NEEDED dependencies on
 * libmkl_intel_lp64, libmkl_sequential, and libmkl_core, and the factory loads
 * the adjacent shim with RTLD_LOCAL in a new glibc link-map namespace. The
 * namespace is required because RTLD_LOCAL alone still permits pre-existing
 * global host symbols to interpose on new dependencies. The components are
 * intrinsically LP64 and sequential, so xtbloom never loads libmkl_rt, calls
 * MKL_Set_Interface_Layer, reads MKL interface-layer state, or mutates an
 * embedding process's MKL state. A missing or invalid shim fails
 * deterministically; MKL never falls back to the base namespace. Plain LP64
 * Linux wheels apply the same namespace isolation to a hash-verified private
 * shim loaded by absolute sibling path. auditwheel vendors and collision-
 * renames the shim's scipy-openblas32 dependency closure. macOS and Windows
 * instead load a renamed provider by absolute sibling path; that is a private
 * payload boundary but not Linux-style link-map isolation. The upstream Python
 * distribution is a build input only and is never imported or required at
 * runtime. Pyodide wheels use the official content-pinned WebAssembly
 * OpenBLAS artifact and a narrow LAPACKE adapter. Because Emscripten has no
 * isolated namespace or deep binding, the Python loader supplies exact
 * installed paths and the adapter resolves raw functions only from that
 * provider handle, never through global SciPy/NumPy symbols. Native system
 * OpenBLAS remains a separate production provider. The testing factory is
 * kept in this internal namespace so tests can install
 * spies and deterministic LAPACK failures without making ABI claims on behalf
 * of an external provider.
 */
class CpuLinearAlgebraBackend {
 public:
  CpuLinearAlgebraBackend() noexcept = default;

  [[nodiscard]] bool ready() const noexcept;
  /* True for any verified lazily-loaded production backend (MKL or OpenBLAS). */
  [[nodiscard]] bool production() const noexcept;
  /* True only when the loaded production backend is the isolated MKL shim. */
  [[nodiscard]] bool production_mkl() const noexcept;
  /* True only for the host-isolated MKL shim provider, which never mutates the
   * embedding process's MKL interface/threading state. */
  [[nodiscard]] bool production_mkl_isolated() const noexcept;
  /* True only for the private OpenBLAS cohort bundled in Linux wheels and
   * loaded in its own glibc link-map namespace. Desktop private providers do
   * not claim this stronger isolation property. */
  [[nodiscard]] bool production_openblas_isolated() const noexcept;

 private:
  enum class Origin : std::uint8_t {
    kNone,
    kMklShimLp64,
    kOpenBlasIsolatedLp64,
    kBundledOpenBlasLp64,
    kOpenBlasLp64,
    kInternalTestLp64,
  };

  Origin origin_ = Origin::kNone;
  LapackDpotrfWork dpotrf_work_ = nullptr;
  LapackDpoconWork dpocon_work_ = nullptr;
  LapackDsyevdWork dsyevd_work_ = nullptr;
  CblasDtrsm dtrsm_ = nullptr;
  CblasDgemm dgemm_ = nullptr;
  BlasSetNumThreadsLocal set_num_threads_local_ = nullptr;

  friend xtbloom_status_t make_mkl_rt_lp64_backend(CpuLinearAlgebraBackend& backend,
                                                   std::string& error);
  friend xtbloom_status_t make_internal_test_lp64_backend(
      LapackDpotrfWork dpotrf_work, LapackDpoconWork dpocon_work, LapackDsyevdWork dsyevd_work,
      CblasDtrsm dtrsm, CblasDgemm dgemm, BlasSetNumThreadsLocal set_num_threads_local,
      CpuLinearAlgebraBackend& backend, std::string& error);
  friend struct CpuLinearAlgebraAccess;
};

xtbloom_status_t make_mkl_rt_lp64_backend(CpuLinearAlgebraBackend& backend, std::string& error);

/* Internal test-only dependency injection; production must use the runtime factory. */
xtbloom_status_t make_internal_test_lp64_backend(
    LapackDpotrfWork dpotrf_work, LapackDpoconWork dpocon_work, LapackDsyevdWork dsyevd_work,
    CblasDtrsm dtrsm, CblasDgemm dgemm, BlasSetNumThreadsLocal set_num_threads_local,
    CpuLinearAlgebraBackend& backend, std::string& error);

struct EigensolverPlanData;

/*
 * Compact immutable handle for all topology, electronic, layout, and scratch
 * metadata required by the CPU eigensolver. Copies are O(1), remain cache-
 * compatible, and make hot-path plan validation O(1).
 */
class EigensolverPlan {
 public:
  EigensolverPlan() noexcept = default;
  EigensolverPlan(const EigensolverPlan&) noexcept = default;
  EigensolverPlan(EigensolverPlan&&) noexcept = default;
  EigensolverPlan& operator=(const EigensolverPlan&) noexcept = default;
  EigensolverPlan& operator=(EigensolverPlan&&) noexcept = default;
  ~EigensolverPlan() = default;

  [[nodiscard]] bool sealed() const noexcept;
  [[nodiscard]] std::int64_t batch_size() const noexcept;
  [[nodiscard]] std::int64_t total_matrix_elements() const noexcept;
  [[nodiscard]] std::int64_t maximum_orbitals() const noexcept;
  [[nodiscard]] double minimum_overlap_rcond() const noexcept;
  [[nodiscard]] std::size_t overlap_cache_size_bytes() const noexcept;
  [[nodiscard]] std::size_t worker_workspace_size_bytes() const noexcept;
  [[nodiscard]] std::size_t workspace_size_bytes() const noexcept;
  [[nodiscard]] std::size_t resident_bytes() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& matrix_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& orbital_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int32_t>& spin_channels() const noexcept;
  [[nodiscard]] const std::vector<double>& alpha_electron_counts() const noexcept;
  [[nodiscard]] const std::vector<double>& beta_electron_counts() const noexcept;
  /* True when a byte range aliases this plan's immutable object or backing storage. */
  [[nodiscard]] bool overlaps_storage(const void* data, std::size_t size_bytes) const noexcept;
  [[nodiscard]] const EigensolverPlanData* identity() const noexcept;

 private:
  explicit EigensolverPlan(std::shared_ptr<const EigensolverPlanData> data) noexcept;
  std::shared_ptr<const EigensolverPlanData> data_;

  friend xtbloom_status_t make_eigensolver_plan(const WavefunctionLayout& layout,
                                                EigensolverPlan& plan, std::string& error,
                                                double minimum_overlap_rcond);
};

struct EigensolverOverlapCache {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0u;
  double* cholesky_factors = nullptr;
  std::uint64_t* geometry_generations = nullptr;
  xtbloom_status_t* system_statuses = nullptr;
  const EigensolverPlanData* plan_identity = nullptr;
};

/*
 * One caller-owned numerical workspace descriptor with two binding modes.
 *
 * bind_eigensolver_worker_workspace binds only the leading maximum-system
 * scratch and leaves all staging pointers null. Its size depends only on the
 * largest system, so B parallel workers require O(B*max_system_size) memory.
 * bind_eigensolver_workspace additionally binds factor_staging and batch_*
 * arrays, whose unpublished full-batch results provide call-level atomicity.
 */
struct EigensolverWorkspace {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0u;
  double* coefficients = nullptr;
  double* densities = nullptr;
  double* energy_weighted_densities = nullptr;
  double* eigenvalues = nullptr;
  double* occupations = nullptr;
  double* lapack_work = nullptr;
  LapackInt* lapack_integer_work = nullptr;

  double* factor_staging = nullptr;
  std::uint64_t* factor_generation_staging = nullptr;
  xtbloom_status_t* factor_status_staging = nullptr;

  double* batch_coefficients = nullptr;
  double* batch_densities = nullptr;
  double* batch_energy_weighted_densities = nullptr;
  double* batch_eigenvalues = nullptr;
  double* batch_occupations = nullptr;
  xtbloom_status_t* batch_system_statuses = nullptr;
  double* batch_chemical_potentials = nullptr;
  double* batch_entropies = nullptr;
  double* batch_band_energies = nullptr;
  double* batch_free_energies = nullptr;

  const EigensolverPlanData* plan_identity = nullptr;
};

/*
 * Entropy is dimensionless (units of k_B). Temperature arguments are k_B*T in
 * Hartree, so free_energy = band_energy - temperature*entropy is in Hartree.
 */
struct EigensolverThermodynamicsView {
  xtbloom_status_t* system_statuses = nullptr;
  std::size_t system_status_capacity = 0u;
  double* chemical_potentials = nullptr;
  std::size_t chemical_potential_capacity = 0u;
  double* entropies = nullptr;
  std::size_t entropy_capacity = 0u;
  double* band_energies = nullptr;
  std::size_t band_energy_capacity = 0u;
  double* free_energies = nullptr;
  std::size_t free_energy_capacity = 0u;
};

/*
 * Internal adaptive-SCC recycle policy. This is deliberately not public ABI:
 * production SCC owns the policy and may change its calibration without
 * changing caller-visible descriptors. A recycle attempt always retains the
 * previous complete S-orthonormal basis and falls back to the ordinary dense
 * solve before publishing when any quality gate fails.
 */
struct EigensolverRecyclePolicy {
  std::int64_t minimum_orbitals = 96;
  std::int64_t minimum_full_iterations = 3;
  std::int64_t dense_confirmations_after_recycle = 2;
  std::int64_t fallback_cooldown_iterations = 1;
  std::int64_t minimum_virtual_buffer = 12;
  std::int64_t maximum_expansion_orbitals = 8;
  double virtual_buffer_fraction = 0.20;
  double maximum_active_fraction = 0.85;
  double convergence_guard_factor = 8.0;
  double occupation_tail_tolerance = 1.0e-10;
  double metric_orthogonality_tolerance = 1.0e-8;
  double maximum_residual_backward_error = 1.0e-4;
  double rms_residual_backward_error = 4.0e-5;
  double minimum_boundary_gap = 1.0e-6;
  double maximum_residual_gap_ratio = 2.0e-2;
};

enum class EigensolverSolveMode : std::uint32_t {
  kFull = 0u,
  kRecycled = 1u,
  kRecycleFallback = 2u,
};

/* One-system diagnostic returned without changing numerical semantics. */
struct EigensolverSolveReport {
  EigensolverSolveMode mode = EigensolverSolveMode::kFull;
  std::int64_t active_orbitals = 0;
  double maximum_backward_error = 0.0;
  double rms_backward_error = 0.0;
  double boundary_gap = 0.0;
  double residual_gap_ratio = 0.0;
};

xtbloom_status_t make_eigensolver_plan(const WavefunctionLayout& layout, EigensolverPlan& plan,
                                       std::string& error, double minimum_overlap_rcond = 1.0e-12);

xtbloom_status_t bind_eigensolver_overlap_cache(const EigensolverPlan& plan, void* workspace,
                                                std::size_t workspace_size,
                                                EigensolverOverlapCache& cache, std::string& error);
xtbloom_status_t bind_eigensolver_workspace(const EigensolverPlan& plan, void* workspace,
                                            std::size_t workspace_size, EigensolverWorkspace& view,
                                            std::string& error);
xtbloom_status_t bind_eigensolver_worker_workspace(const EigensolverPlan& plan, void* workspace,
                                                   std::size_t workspace_size,
                                                   EigensolverWorkspace& view, std::string& error);

/*
 * Factor packed symmetric overlaps into persistent column-major Cholesky
 * factors. Structural/backend errors are whole-call failures and publish
 * nothing; positive-definiteness and conditioning failures are recorded per
 * system when the complete staged batch is committed.
 */
xtbloom_status_t factor_overlap_cpu(const EigensolverPlan& plan, const double* overlap,
                                    std::uint64_t geometry_generation,
                                    const CpuLinearAlgebraBackend& backend,
                                    const EigensolverWorkspace& workspace,
                                    const EigensolverOverlapCache& cache, std::string& error);

/*
 * Solve all systems serially into unpublished full-batch staging after one
 * whole-call validation pass. Results are committed only after all systems
 * have ruled out call-level backend failures. The MKL backend temporarily
 * requests one BLAS thread, avoiding nested oversubscription.
 */
xtbloom_status_t solve_eigensystems_cpu(
    const EigensolverPlan& plan, const EigensolverOverlapCache& overlap_cache,
    std::uint64_t geometry_generation, const double* hamiltonians, double temperature,
    const CpuLinearAlgebraBackend& backend, const EigensolverWorkspace& workspace,
    const WavefunctionView& wavefunction, const EigensolverThermodynamicsView& thermodynamics,
    std::string& error);

/*
 * Allocation-free one-system worker primitive. Runtime schedulers may invoke
 * this concurrently for different systems when each worker owns a distinct
 * EigensolverWorkspace and std::string error object. Concurrent calls must
 * also target different systems so their wavefunction and thermodynamic output
 * slices are disjoint. Validation is O(1) plus the fixed number of output
 * fields and never scans other batch members.
 */
xtbloom_status_t solve_eigensystem_cpu(
    const EigensolverPlan& plan, std::int64_t system, const EigensolverOverlapCache& overlap_cache,
    std::uint64_t geometry_generation, const double* system_hamiltonians, double temperature,
    const CpuLinearAlgebraBackend& backend, const EigensolverWorkspace& workspace,
    const WavefunctionView& wavefunction, const EigensolverThermodynamicsView& thermodynamics,
    std::string& error);

/*
 * Allocation-free guarded Rayleigh--Ritz wrapper for one SCC system.
 * request_recycle is only a hint: unsupported layouts, insufficient savings,
 * ill-conditioned subspaces, poor residuals, or frontier ambiguity execute
 * the existing dense solver in the same logical SCC iteration. Accepted
 * updates diagonalize the omitted complement, absorb a bounded set of its
 * lowest current Ritz vectors into the active space, and gate the remaining
 * cross-block residual after the augmented solve. This preserves the current
 * complete-array semantics. A recycle miss is reported through
 * EigensolverSolveReport, never as a numerical system failure. Unrestricted
 * systems intentionally stay on the dense path until spin-major recycle
 * evidence is complete.
 */
xtbloom_status_t solve_eigensystem_adaptive_cpu(
    const EigensolverPlan& plan, std::int64_t system, const EigensolverOverlapCache& overlap_cache,
    std::uint64_t geometry_generation, const double* system_hamiltonians, double temperature,
    const CpuLinearAlgebraBackend& backend, const EigensolverWorkspace& workspace,
    const WavefunctionView& wavefunction, const EigensolverThermodynamicsView& thermodynamics,
    const EigensolverRecyclePolicy& policy, bool request_recycle, EigensolverSolveReport& report,
    std::string& error);

/* Standalone tblite-compatible per-spin Aufbau/Fermi filling helper. */
xtbloom_status_t fill_occupations_cpu(std::int64_t orbital_count, const double* eigenvalues,
                                      double electron_count, double temperature,
                                      double* occupations, double& chemical_potential,
                                      double& entropy, std::string& error);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_EIGENSOLVER_HPP
