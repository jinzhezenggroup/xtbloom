#ifndef XTBLOOM_RUNTIME_GFN2_CPU_EXECUTION_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_RUNTIME_GFN2_CPU_EXECUTION_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cpu_dispatch/features.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail {

struct Gfn2CpuPeriodicSnapshot;

/*
 * Context-owned cache for restricted host GFN2 execution.
 *
 * The implementation keeps immutable per-system topology plans and their
 * caller-owned numerical workspaces alive across repeated public C API calls.
 * Geometry-dependent caches and SCC state are refreshed for every inference.
 * A fully converged inference also retains a per-system electronic checkpoint
 * so a strict WARM start can seed the next SCC run from the converged state
 * instead of resuming from the SAD guess.
 */
class Gfn2CpuExecutionCache {
 public:
  /*
   * cpu_threads is the context-wide outer batch parallelism requested by the
   * public API. Zero selects an affinity-aware automatic value; one keeps the
   * execution path serial. The implementation owns persistent workers so a
   * steady-state compute call never creates or destroys threads.
   */
  explicit Gfn2CpuExecutionCache(std::int32_t cpu_threads, CpuIsa cpu_isa = CpuIsa::kBaseline);
  ~Gfn2CpuExecutionCache();

  Gfn2CpuExecutionCache(const Gfn2CpuExecutionCache&) = delete;
  Gfn2CpuExecutionCache& operator=(const Gfn2CpuExecutionCache&) = delete;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend xtbloom_status_t execute_restricted_gfn2_cpu(Gfn2CpuExecutionCache& cache,
                                                      const xtbloom_batch_t& batch,
                                                      const xtbloom_compute_options_t& options,
                                                      xtbloom_batch_result_t& result,
                                                      std::string& error);
  friend xtbloom_status_t prepare_restricted_gfn2_cpu(Gfn2CpuExecutionCache& cache,
                                                      const xtbloom_batch_t& batch,
                                                      const xtbloom_compute_options_t& options,
                                                      bool& reused, std::string& error);
  friend std::size_t persistent_workspace_bytes_restricted_gfn2_cpu(
      Gfn2CpuExecutionCache& cache) noexcept;
  friend xtbloom_status_t snapshot_restricted_gfn2_periodic_state(Gfn2CpuExecutionCache& cache,
                                                                  Gfn2CpuPeriodicSnapshot& snapshot,
                                                                  std::string& error);
};

/*
 * Read-only state exported to the CUDA periodic bridge after a completed CPU
 * SCC attempt.  The vectors are deliberately value-owned: the CPU cache may
 * start another request only after the bridge has finished consuming this
 * snapshot, and no CUDA work can retain pointers into a mutable SystemExecution.
 */
struct Gfn2CpuPeriodicSystemSnapshot {
  bool native_periodic = false;
  xtbloom_status_t status = XTBLOOM_STATUS_INTERNAL_ERROR;
  std::vector<double> shell_charges;
  std::vector<double> coordination_numbers;
  std::vector<double> atomic_charges;
  std::vector<double> atomic_dipoles;
  std::vector<double> atomic_quadrupoles;

  std::vector<double> ewald_matrix;
  std::vector<double> ewald_shell_potentials;
  std::vector<double> ewald_energies;
  std::vector<double> ewald_gradients;
  std::vector<double> ewald_strain;

  std::vector<double> multipole_charge_dipole;
  std::vector<double> multipole_dipole_dipole;
  std::vector<double> multipole_charge_quadrupole;
  std::vector<double> multipole_charge_potentials;
  std::vector<double> multipole_dipole_potentials;
  std::vector<double> multipole_quadrupole_potentials;
  std::vector<double> multipole_energies;
  std::vector<double> multipole_gradients;
  std::vector<double> multipole_strain;
  std::vector<double> multipole_coordination_adjoint;
};

struct Gfn2CpuPeriodicSnapshot {
  std::vector<Gfn2CpuPeriodicSystemSnapshot> systems;
};

/* Capture the converged/terminal native-periodic state without exposing the
 * CPU cache's private SystemExecution objects to CUDA code. */
xtbloom_status_t snapshot_restricted_gfn2_periodic_state(Gfn2CpuExecutionCache& cache,
                                                         Gfn2CpuPeriodicSnapshot& snapshot,
                                                         std::string& error);

/*
 * Execute one already descriptor-validated host request.
 *
 * Inputs are copied before numerical work so under-aligned C buffers remain
 * well-defined and caller mutations cannot race an in-flight synchronous
 * call. Requested outputs and result flags are committed only after every
 * batch member reaches either a successful or documented terminal state.
 */
xtbloom_status_t execute_restricted_gfn2_cpu(Gfn2CpuExecutionCache& cache,
                                             const xtbloom_batch_t& batch,
                                             const xtbloom_compute_options_t& options,
                                             xtbloom_batch_result_t& result, std::string& error);

/*
 * Allocation-permitted fixed-topology setup for a public plan.
 *
 * Stages and validates the request and builds (or reuses) the per-system
 * SystemExecution objects for the requested identity, leaving the cache warm
 * so the following xtbloom_plan_compute runs allocation-free. `reused` is true
 * when the cache already held an identical identity and no system was rebuilt.
 */
xtbloom_status_t prepare_restricted_gfn2_cpu(Gfn2CpuExecutionCache& cache,
                                             const xtbloom_batch_t& batch,
                                             const xtbloom_compute_options_t& options, bool& reused,
                                             std::string& error);

/*
 * Topology- and spin-dependent persistent host reservation (per-system
 * storage plus the copied input request), independent of the requested
 * property flags. Returned value is used by fixed-topology plan queries so
 * their result stays correct even after another topology replaced the shared
 * cache's prepared systems.
 */
std::size_t persistent_workspace_bytes_restricted_gfn2_cpu(Gfn2CpuExecutionCache& cache) noexcept;

#if defined(XTBLOOM_CPU_WORKER_TEARDOWN_TESTING)
/* Test-only observability for the standalone public-runtime teardown binary.
 * These functions are not compiled into the production shared library. */
using Gfn2CpuWorkerTssHook = void (*)(bool after_scc_iteration) noexcept;
/* Invoke hook on a persistent background worker immediately before and after
 * each production SCC iteration containing the generalized eigensolver. */
void set_gfn2_cpu_worker_tss_hook(Gfn2CpuWorkerTssHook hook) noexcept;
void reset_gfn2_cpu_worker_teardown_test_counters() noexcept;
std::size_t gfn2_cpu_test_background_eigensolver_runs() noexcept;
std::size_t gfn2_cpu_test_background_thread_cleanups() noexcept;
bool gfn2_cpu_test_provider_requires_thread_cleanup() noexcept;
#endif

}  // namespace xtbloom::detail

#endif  // XTBLOOM_RUNTIME_GFN2_CPU_EXECUTION_HPP
