#ifndef XTBLOOM_MODEL_GFN1_SCC_DRIVER_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN1_SCC_DRIVER_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "model/gfn1/es2.hpp"
#include "model/gfn1/es3.hpp"
#include "model/gfn1/mulliken.hpp"
#include "model/gfn1/scc_mixer.hpp"
#include "model/gfn1/spin.hpp"
#include "model/gfn1/wavefunction.hpp"
#include "model/gfn2/eigensolver.hpp"
#include "model/gfn2/periodic_embedding.hpp"

namespace xtbloom::detail::gfn1 {

inline constexpr std::size_t kSccDriverWorkspaceAlignment = 64u;
inline constexpr double kDefaultSccEnergyTolerance = 1.0e-8;

using EigensolverPlan = gfn2::EigensolverPlan;
using EigensolverOverlapCache = gfn2::EigensolverOverlapCache;
using EigensolverWorkspace = gfn2::EigensolverWorkspace;
using EigensolverThermodynamicsView = gfn2::EigensolverThermodynamicsView;
using CpuLinearAlgebraBackend = gfn2::CpuLinearAlgebraBackend;
using PeriodicEmbeddingPlan = gfn2::PeriodicEmbeddingPlan;
using PeriodicEmbeddingPlanData = gfn2::PeriodicEmbeddingPlanData;
using PeriodicEmbeddingWorkspace = gfn2::PeriodicEmbeddingWorkspace;

struct SccDriverPlanData;

/* Model-neutral eigensolver projections over GFN1's scalar wavefunction. */
[[nodiscard]] gfn2::EigensolverWavefunctionLayout make_eigensolver_wavefunction_layout(
    const WavefunctionLayout& layout) noexcept;
[[nodiscard]] gfn2::EigensolverWavefunctionView make_eigensolver_wavefunction_view(
    const WavefunctionView& view) noexcept;

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
  [[nodiscard]] bool periodic_embedding_enabled() const noexcept;
  [[nodiscard]] std::size_t state_size_bytes() const noexcept;
  [[nodiscard]] std::size_t workspace_size_bytes() const noexcept;
  /* Heap storage retained by the immutable driver metadata and its copied
   * model-owned layouts. Opaque subplans share their backing allocations with
   * the composing executor and are intentionally not counted twice. */
  [[nodiscard]] std::size_t resident_bytes() const noexcept;
  [[nodiscard]] const SccDriverPlanData* identity() const noexcept;

 private:
  explicit SccDriverPlan(std::shared_ptr<const SccDriverPlanData> data) noexcept;
  std::shared_ptr<const SccDriverPlanData> data_;
  friend xtbloom_status_t make_scc_driver_plan(
      const WavefunctionLayout&, const MullikenPlan&, const ES2Plan&, const ES3Plan&,
      const SpinPolarizationPlan&, const EigensolverPlan&, const SccMixerPlan&,
      const PeriodicEmbeddingPlan*, std::uint64_t, double, double, SccDriverPlan&, std::string&);
};

struct SccDriverGeometryView {
  const double* h0 = nullptr;
  std::int64_t h0_elements = 0;
  MullikenIntegralView integrals;
  ES2GeometryCache es2_cache;
  std::uint64_t geometry_generation = 0u;
  const double* explicit_point_charge_shell_potential = nullptr;
  std::int64_t explicit_point_charge_shell_elements = 0;
  const double* periodic_shifts = nullptr;
  std::int64_t periodic_shift_elements = 0;
  const double* periodic_response_matrices = nullptr;
  std::int64_t periodic_response_elements = 0;
  std::uint64_t periodic_embedding_generation = 0u;
  const PeriodicEmbeddingPlanData* periodic_plan_identity = nullptr;
};

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
  double* spin_energies = nullptr;
  double* explicit_point_charge_energies = nullptr;
  double* periodic_embedding_energies = nullptr;
  double* internal_energies = nullptr;
  std::uint64_t* iterations = nullptr;
  xtbloom_status_t* system_statuses = nullptr;
  std::uint8_t* initialized = nullptr;
  std::uint8_t* converged = nullptr;
  const SccDriverPlanData* plan_identity = nullptr;
};

struct SccDriverWorkspace {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0u;
  WavefunctionView staged_wavefunction;
  double* hamiltonian = nullptr;
  double* shell_charges = nullptr;
  double* atomic_charges = nullptr;
  /* Topology-major ES3 output, disjoint from atomic_charges by contract. */
  double* component_atomic_potential = nullptr;
  double* component_shell_potential = nullptr;
  double* shell_potentials = nullptr;
  double* spin_shell_potentials = nullptr;
  double* raw_qsh = nullptr;
  double* raw_qat = nullptr;
  double* core_energies = nullptr;
  double* es2_energies = nullptr;
  double* es3_energies = nullptr;
  double* spin_energies = nullptr;
  double* explicit_point_charge_energies = nullptr;
  double* periodic_embedding_energies = nullptr;
  double* internal_energies = nullptr;
  double* free_energies = nullptr;
  double* periodic_atomic_potentials = nullptr;
  xtbloom_status_t* periodic_system_statuses = nullptr;
  std::uint8_t* active_systems = nullptr;
  ES2Workspace es2_workspace;
  MullikenWorkspace mulliken_workspace;
  PeriodicEmbeddingWorkspace periodic_embedding_workspace;
  EigensolverWorkspace eigensolver_workspace;
  EigensolverThermodynamicsView thermodynamics;
  SccMixerState staged_mixer_state;
  SccMixerWorkspace mixer_workspace;
  const SccDriverPlanData* plan_identity = nullptr;
};

/*
 * Topology-major stationary state consumed by the analytic-force composer.
 *
 * The converged wavefunction keeps alpha/beta matrices and per-system packed
 * charge/magnetization populations.  Force kernels instead consume one
 * spin-summed matrix field plus an optional alpha-minus-beta response field.
 * Callers own every output below; projection is allocation-free and publishes
 * only after all bindings and numerical inputs have been validated.
 */
struct SccStationaryProjection {
  double* density = nullptr;
  double* energy_weighted_density = nullptr;
  double* spin_density = nullptr;
  std::int64_t matrix_elements = 0;
  double* shell_charges = nullptr;
  double* atomic_charges = nullptr;
  double* scalar_shell_potentials = nullptr;
  double* spin_shell_potentials = nullptr;
  std::int64_t shell_elements = 0;
  std::int64_t atom_elements = 0;
};

xtbloom_status_t make_scc_driver_plan(
    const WavefunctionLayout& wavefunction, const MullikenPlan& mulliken, const ES2Plan& es2,
    const ES3Plan& es3, const SpinPolarizationPlan& spin, const EigensolverPlan& eigensolver,
    const SccMixerPlan& mixer, const PeriodicEmbeddingPlan* periodic_embedding,
    std::uint64_t maximum_iterations, double electronic_temperature, double energy_tolerance,
    SccDriverPlan& plan, std::string& error);

xtbloom_status_t bind_scc_driver_state(const SccDriverPlan& plan, void* workspace,
                                       std::size_t workspace_size, SccDriverState& state,
                                       std::string& error);
xtbloom_status_t bind_scc_driver_workspace(const SccDriverPlan& plan, void* workspace,
                                           std::size_t workspace_size, SccDriverWorkspace& view,
                                           std::string& error);
xtbloom_status_t initialize_scc_driver_state_cpu(const SccDriverPlan& plan,
                                                 const WavefunctionView& wavefunction,
                                                 const SccMixerState& mixer_state,
                                                 const SccDriverState& state,
                                                 std::string& error);
xtbloom_status_t restart_scc_driver_system_cpu(const SccDriverPlan& plan, std::int64_t system,
                                               const WavefunctionView& wavefunction,
                                               const SccMixerState& mixer_state,
                                               const SccDriverState& state, std::string& error);

/* Advance all active systems by one allocation-free, per-system transaction. */
xtbloom_status_t iterate_scc_driver_batch_cpu(
    const SccDriverPlan& plan, const SccDriverGeometryView& geometry,
    const CpuLinearAlgebraBackend& backend, const EigensolverOverlapCache& overlap_cache,
    const WavefunctionView& wavefunction, const SccMixerState& mixer_state,
    const SccDriverState& state, const SccDriverWorkspace& workspace, std::string& error);

/*
 * Rebuild the scalar and optional magnetization shell potentials at an
 * already converged wavefunction without advancing SCC or changing the
 * driver/mixer state.  This is the production seam between SCC and analytic
 * force composition; outputs remain in the supplied canonical driver
 * workspace and no caller wavefunction bytes are modified.
 */
xtbloom_status_t rebuild_scc_stationary_potentials_cpu(
    const SccDriverPlan& plan, const SccDriverGeometryView& geometry,
    const WavefunctionView& wavefunction, const SccDriverWorkspace& workspace,
    std::string& error);

/*
 * Project a converged SCC state into the stationary force representation.
 * packed_shell_potentials follows wavefunction.qsh's per-system
 * charge/magnetization layout.  Restricted members contribute zero to the
 * batch-wide spin outputs when an unrestricted peer requires those buffers.
 */
xtbloom_status_t project_scc_stationary_state_cpu(
    const WavefunctionLayout& layout, const WavefunctionView& wavefunction,
    const double* packed_shell_potentials, std::int64_t packed_shell_potential_elements,
    const SccStationaryProjection& projection, std::string& error);

}  // namespace xtbloom::detail::gfn1

#endif  // XTBLOOM_MODEL_GFN1_SCC_DRIVER_HPP
