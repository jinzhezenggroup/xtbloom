#ifndef GPUXTB_MODEL_GFN2_SCC_DRIVER_HPP
#define GPUXTB_MODEL_GFN2_SCC_DRIVER_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "gpuxtb/gpuxtb.h"
#include "model/gfn2/aes2.hpp"
#include "model/gfn2/eigensolver.hpp"
#include "model/gfn2/es2.hpp"
#include "model/gfn2/es3.hpp"
#include "model/gfn2/mulliken.hpp"
#include "model/gfn2/scc_mixer.hpp"
#include "model/gfn2/wavefunction.hpp"

namespace gpuxtb::detail::gfn2 {

inline constexpr std::size_t kSccDriverWorkspaceAlignment = 64u;

struct SccDriverPlanData;

/*
 * Immutable orchestration metadata for a restricted, nonperiodic,
 * electrostatic-only GFN2 SCC scaffold.
 *
 * The current CPU implementation deliberately prepares the classical
 * potentials and Mulliken populations as full-batch operations because those
 * component primitives publish atomically only at batch granularity. The
 * generalized eigensolve and mixer are still invoked only for active systems.
 * This preserves skip-converged behavior without duplicating the physical
 * kernels. A future per-system Mulliken primitive can reuse this plan and
 * split the prepare/finalize barriers without changing the state model.
 * Self-consistent D4 charge potentials are not yet included; therefore this
 * scaffold is not a complete GFN2-xTB inference path and must not be used as a
 * tblite-equivalent molecular oracle until issue #27 is integrated.
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
  [[nodiscard]] std::size_t state_size_bytes() const noexcept;
  [[nodiscard]] std::size_t workspace_size_bytes() const noexcept;
  [[nodiscard]] const SccDriverPlanData* identity() const noexcept;

 private:
  explicit SccDriverPlan(std::shared_ptr<const SccDriverPlanData> data) noexcept;
  std::shared_ptr<const SccDriverPlanData> data_;

  friend gpuxtb_status_t make_scc_driver_plan(
      const WavefunctionLayout& wavefunction, const MullikenPlan& mulliken, const ES2Plan& es2,
      const ES3Plan& es3, const AES2Plan& aes2, const EigensolverPlan& eigensolver,
      const SccMixerPlan& mixer, std::uint64_t maximum_iterations, double electronic_temperature,
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
 * Periodic b+Aq embedding is intentionally not duplicated here. The driver
 * will consume #35's validated PeriodicEmbedding potential view when that
 * component is available.
 */
struct SccDriverGeometryView {
  const double* h0 = nullptr;
  std::int64_t h0_elements = 0;
  MullikenIntegralView integrals;
  ES2GeometryCache es2_cache;
  AES2GeometryCache aes2_cache;
  std::uint64_t geometry_generation = 0u;

  const double* explicit_point_charge_shell_potential = nullptr;
  std::int64_t explicit_point_charge_shell_elements = 0;
};

/* Persistent, caller-owned driver status and scalar trace. */
struct SccDriverState {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0u;

  double* free_energies = nullptr;
  double* previous_free_energies = nullptr;
  double* free_energy_changes = nullptr;
  double* entropies = nullptr;
  double* band_energies = nullptr;
  std::uint64_t* iterations = nullptr;
  gpuxtb_status_t* system_statuses = nullptr;
  std::uint8_t* initialized = nullptr;
  std::uint8_t* converged = nullptr;

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
  double* raw_qsh = nullptr;
  double* raw_qat = nullptr;
  double* raw_dipoles = nullptr;
  double* raw_quadrupoles = nullptr;
  std::uint8_t* active_systems = nullptr;

  ES2Workspace es2_workspace;
  AES2Workspace aes2_workspace;
  MullikenWorkspace mulliken_workspace;
  EigensolverWorkspace eigensolver_workspace;
  EigensolverThermodynamicsView thermodynamics;
  SccMixerState staged_mixer_state;
  SccMixerWorkspace mixer_workspace;

  const SccDriverPlanData* plan_identity = nullptr;
};

/*
 * Seal exact component compatibility and precompute all state/scratch offsets.
 * Unrestricted layouts are rejected until a spin-polarization potential is
 * available. Periodic embedding and self-consistent D4 are also not composed
 * yet. Stored free-energy values are only a provisional eigensolver electronic
 * trace (band Helmholtz free energy), not the complete SCC total energy, and
 * do not enter convergence. Complete-energy convergence will be added once all
 * interaction energies provide per-system failure isolation.
 */
gpuxtb_status_t make_scc_driver_plan(const WavefunctionLayout& wavefunction,
                                     const MullikenPlan& mulliken, const ES2Plan& es2,
                                     const ES3Plan& es3, const AES2Plan& aes2,
                                     const EigensolverPlan& eigensolver, const SccMixerPlan& mixer,
                                     std::uint64_t maximum_iterations,
                                     double electronic_temperature, SccDriverPlan& plan,
                                     std::string& error);

gpuxtb_status_t bind_scc_driver_state(const SccDriverPlan& plan, void* workspace,
                                      std::size_t workspace_size, SccDriverState& state,
                                      std::string& error);

gpuxtb_status_t bind_scc_driver_workspace(const SccDriverPlan& plan, void* workspace,
                                          std::size_t workspace_size, SccDriverWorkspace& view,
                                          std::string& error);

/* Initialize both the mixer history and driver trace as one logical action. */
gpuxtb_status_t initialize_scc_driver_state_cpu(const SccDriverPlan& plan,
                                                const WavefunctionView& wavefunction,
                                                const SccMixerState& mixer_state,
                                                const SccDriverState& state, std::string& error);

/* Restart one system from its current public multipoles (raw when converged). */
gpuxtb_status_t restart_scc_driver_system_cpu(const SccDriverPlan& plan, std::int64_t system,
                                              const WavefunctionView& wavefunction,
                                              const SccMixerState& mixer_state,
                                              const SccDriverState& state, std::string& error);

/*
 * Advance every active system by one SCC iteration.
 *
 * Structural/binding/backend contract failures publish nothing. Per-system
 * eigensolver, mixer, and max-iteration failures are different: peers keep
 * running and their successful results are committed, then the call returns
 * the first non-success per-system status. Callers therefore must not
 * interpret such a return as an uncommitted batch. Converged and terminal
 * systems are skipped. A converged system publishes density-derived raw
 * Mulliken multipoles for subsequent total-energy/force evaluation; the
 * mixer retains its private next-input vector until a restart reinitializes it
 * from that public raw state. Successful steady-state calls perform no dynamic
 * allocation.
 */
gpuxtb_status_t iterate_scc_driver_batch_cpu(
    const SccDriverPlan& plan, const SccDriverGeometryView& geometry,
    const CpuLinearAlgebraBackend& backend, const EigensolverOverlapCache& overlap_cache,
    const WavefunctionView& wavefunction, const SccMixerState& mixer_state,
    const SccDriverState& state, const SccDriverWorkspace& workspace, std::string& error);

}  // namespace gpuxtb::detail::gfn2

#endif  // GPUXTB_MODEL_GFN2_SCC_DRIVER_HPP
