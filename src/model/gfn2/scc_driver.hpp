#ifndef XTBLOOM_MODEL_GFN2_SCC_DRIVER_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_SCC_DRIVER_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "model/gfn2/aes2.hpp"
#include "model/gfn2/d4.hpp"
#include "model/gfn2/eigensolver.hpp"
#include "model/gfn2/es2.hpp"
#include "model/gfn2/es3.hpp"
#include "model/gfn2/mulliken.hpp"
#include "model/gfn2/parallel_executor.hpp"
#include "model/gfn2/periodic_embedding.hpp"
#include "model/gfn2/scc_mixer.hpp"
#include "model/gfn2/spin.hpp"
#include "model/gfn2/wavefunction.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {

inline constexpr std::size_t kSccDriverWorkspaceAlignment = 64u;
inline constexpr double kDefaultSccEnergyTolerance = 1.0e-8;

struct SccDriverPlanData;

/*
 * Immutable orchestration metadata for a restricted, electrostatic-only GFN2
 * SCC scaffold with optional self-consistent D4 and periodic charge embedding.
 *
 * The CPU driver prepares the classical potentials and Mulliken
 * Hamiltonian/population per active system only: converged, terminal, and
 * failed peers perform no classical or Mulliken arithmetic. A per-system
 * numerical failure during potential, Hamiltonian, or population assembly is
 * data-level and leaves successful peers commit-ready, matching the semantics
 * of eigensolver, mixer, and periodic failures. The generalized eigensolve
 * and mixer are invoked only for active systems. D4 contributes its
 * charge-dependent two-body atom potential on every active iteration. The
 * charge-independent ATM contribution remains outside SCC and is evaluated as
 * part of the eventual total-energy/gradient path.
 */
class SccDriverPlan {
 public:
  SccDriverPlan() noexcept = default;
  SccDriverPlan(const SccDriverPlan&) noexcept = default;
  SccDriverPlan(SccDriverPlan&&) noexcept = default;
  SccDriverPlan& operator=(const SccDriverPlan&) noexcept = default;
  SccDriverPlan& operator=(SccDriverPlan&&) noexcept = default;
  ~SccDriverPlan() = default;

  [[nodiscard]] bool sealed() const noexcept;
  [[nodiscard]] std::int64_t batch_size() const noexcept;
  [[nodiscard]] std::uint64_t maximum_iterations() const noexcept;
  [[nodiscard]] double electronic_temperature() const noexcept;
  [[nodiscard]] double energy_tolerance() const noexcept;
  [[nodiscard]] bool d4_enabled() const noexcept;
  [[nodiscard]] bool periodic_embedding_enabled() const noexcept;
  [[nodiscard]] std::size_t state_size_bytes() const noexcept;
  [[nodiscard]] std::size_t workspace_size_bytes() const noexcept;
  [[nodiscard]] std::size_t resident_bytes() const noexcept;
  [[nodiscard]] const SccDriverPlanData* identity() const noexcept;

 private:
  explicit SccDriverPlan(std::shared_ptr<const SccDriverPlanData> data) noexcept;
  std::shared_ptr<const SccDriverPlanData> data_;

  friend xtbloom_status_t make_scc_driver_plan(
      const WavefunctionLayout& wavefunction, const MullikenPlan& mulliken, const ES2Plan& es2,
      const ES3Plan& es3, const AES2Plan& aes2, const EigensolverPlan& eigensolver,
      const SccMixerPlan& mixer, std::uint64_t maximum_iterations, double electronic_temperature,
      SccDriverPlan& plan, std::string& error);
  friend xtbloom_status_t make_scc_driver_plan(
      const WavefunctionLayout& wavefunction, const MullikenPlan& mulliken, const ES2Plan& es2,
      const ES3Plan& es3, const AES2Plan& aes2, const EigensolverPlan& eigensolver,
      const SccMixerPlan& mixer, const D4Plan* d4, const PeriodicEmbeddingPlan* periodic_embedding,
      std::uint64_t maximum_iterations, double electronic_temperature, SccDriverPlan& plan,
      std::string& error);
  friend xtbloom_status_t make_scc_driver_plan(
      const WavefunctionLayout& wavefunction, const MullikenPlan& mulliken, const ES2Plan& es2,
      const ES3Plan& es3, const AES2Plan& aes2, const EigensolverPlan& eigensolver,
      const SccMixerPlan& mixer, const D4Plan* d4, const PeriodicEmbeddingPlan* periodic_embedding,
      std::uint64_t maximum_iterations, double electronic_temperature, double energy_tolerance,
      SccDriverPlan& plan, std::string& error);
};

/*
 * Geometry-generation inputs reused by every SCC iteration.
 *
 * h0 stores one spin-independent dense AO matrix per system in MullikenPlan's
 * matrix packing. The driver copies it into every spin channel before adding
 * SCC shifts. explicit_point_charge_shell_potential is the already-cached
 * shell potential V^PC used in every iteration; a null pointer with zero
 * elements disables explicit point charges.
 * If the plan enables D4, d4_cache must be current for geometry_generation and
 * belong to the exact D4Plan sealed into the driver. All D4 cache fields must
 * remain null/zero when D4 is disabled.
 * If the plan enables periodic embedding, periodic_shifts and
 * periodic_response_matrices contain b and the packed dense symmetric A from
 * PeriodicEmbeddingPlan. periodic_embedding_generation is an independent,
 * nonzero host epoch because the environment may change while the QM geometry
 * and its ES2/AES2 caches remain fixed. The plan identity prevents same-sized
 * data from another topology being accepted. All periodic fields must remain
 * null/zero when the component is disabled.
 */
struct SccDriverGeometryView {
  const double* h0 = nullptr;
  std::int64_t h0_elements = 0;
  MullikenIntegralView integrals;
  ES2GeometryCache es2_cache;
  AES2GeometryCache aes2_cache;
  D4GeometryCache d4_cache;
  std::uint64_t geometry_generation = 0u;

  const double* explicit_point_charge_shell_potential = nullptr;
  std::int64_t explicit_point_charge_shell_elements = 0;

  const double* periodic_shifts = nullptr;
  std::int64_t periodic_shift_elements = 0;
  const double* periodic_response_matrices = nullptr;
  std::int64_t periodic_response_elements = 0;
  std::uint64_t periodic_embedding_generation = 0u;
  const PeriodicEmbeddingPlanData* periodic_plan_identity = nullptr;

  /* Uniform external electric field (pilot interaction, tag 0x0101).
   *
   * field_atomic_potential holds the per-atom scalar potential
   * vat_i = -E . r_i (one double per atom across the whole batch) and
   * field_dipole_potential holds the per-atom dipolar potential vdp = -E
   * (three doubles per atom across the whole batch), in atomic units. Both
   * arrays are caller-owned and geometry-dependent; they must either both be
   * present (fields are then added to the charge-channel atomic and dipolar
   * SCC potentials every iteration) or both be null, preserving the
   * field-free path byte-for-byte. The energy term -sum_i q_i (E . r_i)
   * - sum_i E . d_i is folded into the SCC energy trace. The stationary
   * response follows from these potentials, while the remaining explicit
   * coordinate force +q_i E is added at the stationary force boundary. */
  const double* field_atomic_potential = nullptr;
  std::int64_t field_atomic_potential_elements = 0;
  const double* field_dipole_potential = nullptr;
  std::int64_t field_dipole_potential_elements = 0;
};

/*
 * Persistent, caller-owned driver status and scalar trace.
 *
 * Every component trace is evaluated from the density-derived raw multipoles
 * of the most recently successful iteration. internal_energies is Tr(P H0)
 * plus all SCC interaction terms; free_energies additionally includes the
 * finite-temperature -kT*S contribution and is the energy convergence trace.
 * D4 ATM and geometry-only repulsion remain outside SCC. Optional component
 * pointers are null when their plan is disabled.
 */
struct SccDriverState {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0u;

  double* free_energies = nullptr;
  double* previous_free_energies = nullptr;
  double* free_energy_changes = nullptr;
  double* entropies = nullptr;
  double* band_energies = nullptr;
  double* core_energies = nullptr;
  double* es2_energies = nullptr;
  double* es3_energies = nullptr;
  double* aes2_energies = nullptr;
  double* spin_energies = nullptr;
  double* d4_two_body_energies = nullptr;
  double* explicit_point_charge_energies = nullptr;
  double* periodic_embedding_energies = nullptr;
  double* internal_energies = nullptr;
  std::uint64_t* iterations = nullptr;
  /* Committed eigensolver provenance and diagnostics. A recycled SCC
   * iteration is followed by dense correction/confirmation before terminal
   * convergence may publish. Counts include only complete per-system SCC
   * transactions, so they never advance ahead of the public wavefunction. */
  std::uint64_t* eigensolver_geometry_generations = nullptr;
  std::uint64_t* full_eigensolves = nullptr;
  std::uint64_t* recycled_eigensolves = nullptr;
  std::uint64_t* recycle_fallbacks = nullptr;
  EigensolverSolveMode* last_eigensolver_modes = nullptr;
  xtbloom_status_t* system_statuses = nullptr;
  std::uint8_t* initialized = nullptr;
  std::uint8_t* converged = nullptr;
  std::uint8_t* dense_confirmations_remaining = nullptr;
  std::uint8_t* recycle_cooldowns_remaining = nullptr;

  const SccDriverPlanData* plan_identity = nullptr;
};

/*
 * Caller-owned unpublished numerical storage.
 *
 * Public leading pointers are diagnostics only. Nested descriptors are exact
 * bindings created by bind_scc_driver_workspace and must not be modified.
 * The staged wavefunction prevents any structural or classical full-batch
 * failure from publishing partial orbital or multipole results.
 */
struct SccDriverWorkspace {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0u;

  WavefunctionView staged_wavefunction;
  double* hamiltonian = nullptr;
  double* shell_charges = nullptr;
  double* atomic_charges = nullptr;
  double* atomic_dipoles = nullptr;
  double* atomic_quadrupoles = nullptr;
  double* component_shell_potential = nullptr;
  double* component_atomic_potential = nullptr;
  double* component_dipole_potential = nullptr;
  double* component_quadrupole_potential = nullptr;
  double* atomic_potentials = nullptr;
  double* shell_potentials = nullptr;
  double* dipole_potentials = nullptr;
  double* quadrupole_potentials = nullptr;
  double* spin_shell_potentials = nullptr;
  double* raw_qsh = nullptr;
  double* raw_qat = nullptr;
  double* raw_dipoles = nullptr;
  double* raw_quadrupoles = nullptr;
  double* core_energies = nullptr;
  double* es2_energies = nullptr;
  double* es3_energies = nullptr;
  double* aes2_energies = nullptr;
  double* spin_energies = nullptr;
  double* explicit_point_charge_energies = nullptr;
  double* internal_energies = nullptr;
  double* free_energies = nullptr;
  double* periodic_atomic_potentials = nullptr;
  double* periodic_embedding_energies = nullptr;
  xtbloom_status_t* periodic_system_statuses = nullptr;
  double* d4_atomic_potentials = nullptr;
  double* d4_two_body_energies = nullptr;
  std::uint8_t* active_systems = nullptr;
  EigensolverSolveMode* eigensolver_modes = nullptr;

  ES2Workspace es2_workspace;
  AES2Workspace aes2_workspace;
  MullikenWorkspace mulliken_workspace;
  D4Workspace d4_workspace;
  PeriodicEmbeddingWorkspace periodic_embedding_workspace;
  EigensolverWorkspace eigensolver_workspace;
  EigensolverThermodynamicsView thermodynamics;
  SccMixerState staged_mixer_state;
  SccMixerWorkspace mixer_workspace;

  const SccDriverPlanData* plan_identity = nullptr;
};

/*
 * Seal exact component compatibility and precompute all state/scratch offsets.
 * Restricted and unrestricted systems may coexist in one ragged batch; the
 * driver derives the atom-local GFN2 spin-polarization plan from the same
 * canonical basis metadata. Compatibility overloads use
 * kDefaultSccEnergyTolerance. The
 * production convergence gate requires both the mixer RMS residual and the
 * absolute complete SCC free-energy change to be strictly below tolerance.
 */
xtbloom_status_t make_scc_driver_plan(const WavefunctionLayout& wavefunction,
                                      const MullikenPlan& mulliken, const ES2Plan& es2,
                                      const ES3Plan& es3, const AES2Plan& aes2,
                                      const EigensolverPlan& eigensolver, const SccMixerPlan& mixer,
                                      std::uint64_t maximum_iterations,
                                      double electronic_temperature, SccDriverPlan& plan,
                                      std::string& error);

/*
 * Enable self-consistent D4, periodic embedding, or both. A non-null optional
 * plan must be sealed and describe exactly the driver's ragged atom topology.
 * Passing nullptr for both components is equivalent to the compatibility
 * overload above.
 */
xtbloom_status_t make_scc_driver_plan(
    const WavefunctionLayout& wavefunction, const MullikenPlan& mulliken, const ES2Plan& es2,
    const ES3Plan& es3, const AES2Plan& aes2, const EigensolverPlan& eigensolver,
    const SccMixerPlan& mixer, const D4Plan* d4, const PeriodicEmbeddingPlan* periodic_embedding,
    std::uint64_t maximum_iterations, double electronic_temperature, SccDriverPlan& plan,
    std::string& error);

/*
 * Explicit complete-free-energy convergence policy. energy_tolerance is in
 * Hartree and must be finite and positive.
 */
xtbloom_status_t make_scc_driver_plan(
    const WavefunctionLayout& wavefunction, const MullikenPlan& mulliken, const ES2Plan& es2,
    const ES3Plan& es3, const AES2Plan& aes2, const EigensolverPlan& eigensolver,
    const SccMixerPlan& mixer, const D4Plan* d4, const PeriodicEmbeddingPlan* periodic_embedding,
    std::uint64_t maximum_iterations, double electronic_temperature, double energy_tolerance,
    SccDriverPlan& plan, std::string& error);

/*
 * Enable the validated CPU periodic charge response. A non-null pointer must
 * name a sealed plan with exactly the driver's ragged atom partition. Passing
 * nullptr is equivalent to the compatibility overload above.
 */
xtbloom_status_t make_scc_driver_plan(const WavefunctionLayout& wavefunction,
                                      const MullikenPlan& mulliken, const ES2Plan& es2,
                                      const ES3Plan& es3, const AES2Plan& aes2,
                                      const EigensolverPlan& eigensolver, const SccMixerPlan& mixer,
                                      const PeriodicEmbeddingPlan* periodic_embedding,
                                      std::uint64_t maximum_iterations,
                                      double electronic_temperature, SccDriverPlan& plan,
                                      std::string& error);

xtbloom_status_t bind_scc_driver_state(const SccDriverPlan& plan, void* workspace,
                                       std::size_t workspace_size, SccDriverState& state,
                                       std::string& error);

xtbloom_status_t bind_scc_driver_workspace(const SccDriverPlan& plan, void* workspace,
                                           std::size_t workspace_size, SccDriverWorkspace& view,
                                           std::string& error);

/* Initialize both the mixer history and driver trace as one logical action. */
xtbloom_status_t initialize_scc_driver_state_cpu(const SccDriverPlan& plan,
                                                 const WavefunctionView& wavefunction,
                                                 const SccMixerState& mixer_state,
                                                 const SccDriverState& state, std::string& error);

/* Restart one system from its current public multipoles (raw when converged). */
xtbloom_status_t restart_scc_driver_system_cpu(const SccDriverPlan& plan, std::int64_t system,
                                               const WavefunctionView& wavefunction,
                                               const SccMixerState& mixer_state,
                                               const SccDriverState& state, std::string& error);

/*
 * Advance every active system by one SCC iteration.
 *
 * Structural/binding/backend contract failures publish nothing. Per-system
 * energy-assembly, periodic-embedding, eigensolver, mixer, and max-iteration
 * failures are different: peers keep running and their successful results
 * are committed, then the call returns the first non-success per-system
 * status. A periodic failure occurs before an eigensolve attempt and therefore
 * does not increment that system's driver iteration. Callers therefore must not
 * interpret such a return as an uncommitted batch. Converged and terminal
 * systems are skipped. A converged system publishes density-derived raw
 * Mulliken multipoles and complete component/free-energy trace; the
 * mixer retains its private next-input vector until a restart reinitializes it
 * from that public raw state. Successful steady-state calls perform no dynamic
 * allocation.
 */
xtbloom_status_t iterate_scc_driver_batch_cpu(
    const SccDriverPlan& plan, const SccDriverGeometryView& geometry,
    const CpuLinearAlgebraBackend& backend, const EigensolverOverlapCache& overlap_cache,
    const WavefunctionView& wavefunction, const SccMixerState& mixer_state,
    const SccDriverState& state, const SccDriverWorkspace& workspace, std::string& error,
    const SccParallelExecutor* parallel = nullptr);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_SCC_DRIVER_HPP
