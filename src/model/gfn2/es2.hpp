#ifndef XTBLOOM_MODEL_GFN2_ES2_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_ES2_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "model/gfn2/basis.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {

struct ES2PlanData;

/*
 * Model-selected shell-hardness averaging for the isotropic effective-Coulomb
 * kernel. Keeping this choice in the sealed plan lets GFN1 and GFN2 share the
 * same geometry/cache/evaluation implementation without reinterpreting either
 * model's parameter table.
 */
enum class ES2HardnessAverage : std::uint8_t {
  kArithmetic = 0,
  kHarmonic = 1,
};

/*
 * Geometry-independent shell-resolved second-order electrostatics data.
 *
 * shell_hardness contains gamma_s = gamma_element * shell_hubbard_scale in
 * Hartree, in BasisPlan shell order. matrix_offsets packs one dense row-major
 * shell matrix per ragged batch member; an n-shell molecule owns n*n values.
 * Plan construction may allocate, while all geometry/SCC operations below use
 * caller-owned storage and allocate nothing on successful steady-state calls.
 * The numerical data is sealed in opaque immutable shared storage: copying a
 * plan is O(1), copies remain cache-compatible, and external code cannot mutate
 * the topology or parameters after construction.
 */
class ES2Plan {
 public:
  ES2Plan() noexcept = default;
  ES2Plan(const ES2Plan&) noexcept = default;
  ES2Plan(ES2Plan&&) noexcept = default;
  ES2Plan& operator=(const ES2Plan&) noexcept = default;
  ES2Plan& operator=(ES2Plan&&) noexcept = default;
  ~ES2Plan() = default;

  [[nodiscard]] bool sealed() const noexcept;
  [[nodiscard]] std::int64_t batch_size() const noexcept;
  [[nodiscard]] std::int64_t total_atoms() const noexcept;
  [[nodiscard]] std::int64_t total_shells() const noexcept;
  [[nodiscard]] std::int64_t total_matrix_elements() const noexcept;
  [[nodiscard]] std::size_t resident_bytes() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& atom_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& batch_shell_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& atom_shell_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& matrix_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& shell_to_atom() const noexcept;
  [[nodiscard]] const std::vector<double>& shell_hardness() const noexcept;
  [[nodiscard]] ES2HardnessAverage hardness_average() const noexcept;

  /* True when a byte range aliases this plan's immutable object or backing storage. */
  [[nodiscard]] bool overlaps_storage(const void* data, std::size_t size_bytes) const noexcept;

  /* Opaque stable token used only for cache compatibility and diagnostics. */
  [[nodiscard]] const ES2PlanData* identity() const noexcept;

 private:
  explicit ES2Plan(std::shared_ptr<const ES2PlanData> data) noexcept;

  std::shared_ptr<const ES2PlanData> data_;

  friend xtbloom_status_t make_es2_plan(const BasisPlan& basis, const std::int32_t* atomic_numbers,
                                        ES2Plan& plan, std::string& error);
  friend xtbloom_status_t make_es2_plan_from_shell_hardness(
      const BasisPlan& basis, ES2HardnessAverage average, const double* shell_hardness,
      std::int64_t shell_hardness_count, ES2Plan& plan, std::string& error);
};

/*
 * Non-owning geometry cache view. The exact plan_identity token prevents a
 * same-extent matrix from another topology from being reused accidentally.
 * At least one copy of the originating plan must remain alive while the cache
 * is used. The descriptor remains trivial and standard-layout so a future
 * CUDA/ROCm backend can use the same flat matrix storage.
 */
struct ES2GeometryCache {
  double* coulomb_matrix = nullptr;
  std::int64_t matrix_elements = 0;
  std::uint64_t geometry_generation = 0;
  const ES2PlanData* plan_identity = nullptr;
};

/*
 * Caller-owned scratch used to preserve whole-batch failure atomicity without
 * recomputing an ES2 quantity. Counts are numbers of doubles, not bytes.
 * The descriptor is POD and may point to ordinary host, pinned, or backend-
 * specific allocations as appropriate for the routine consuming it.
 */
struct ES2Workspace {
  double* matrix_scratch = nullptr;
  std::int64_t matrix_elements = 0;
  double* shell_scratch = nullptr;
  std::int64_t shell_elements = 0;
  double* batch_scratch = nullptr;
  std::int64_t batch_elements = 0;
  double* gradient_scratch = nullptr;
  std::int64_t gradient_elements = 0;
};

/* Build a reusable GFN2 arithmetic-hardness, gexp=2 ES2 plan. */
xtbloom_status_t make_es2_plan(const BasisPlan& basis, const std::int32_t* atomic_numbers,
                               ES2Plan& plan, std::string& error);

/*
 * Seal already expanded model-specific shell hardnesses into the shared ES2
 * kernel. Model builders remain responsible for validating their own element
 * and shell metadata before calling this lower-level constructor.
 */
xtbloom_status_t make_es2_plan_from_shell_hardness(
    const BasisPlan& basis, ES2HardnessAverage average, const double* shell_hardness,
    std::int64_t shell_hardness_count, ES2Plan& plan, std::string& error);

/*
 * Overwrite caller-owned matrix_storage and bind cache to it. Positions use
 * atom-major xyz coordinates in bohr. For shells s,t on different atoms,
 *
 *   Gamma_st = [ R_AB^2 + gamma_st^(-2) ]^(-1/2),
 *   gamma_st = plan-selected average(gamma_s, gamma_t),
 *
 * while every same-atom element is Gamma_st = gamma_st. Gamma is in Hartree.
 * The ordinary GFN2 builder selects arithmetic averaging; GFN1 selects the
 * harmonic average through make_es2_plan_from_shell_hardness.
 * matrix_storage and workspace.matrix_scratch must each contain at least
 * plan.total_matrix_elements() doubles. They must not overlap one another,
 * positions, or immutable plan storage; an existing active cache must not
 * alias the scratch or plan storage. geometry_generation is an opaque caller
 * sequence number for the supplied positions. Active input/output/cache
 * backing buffers must not alias the plan, cache, or workspace descriptor
 * objects themselves. On failure, matrix_storage and cache are unchanged;
 * workspace scratch contents are unspecified.
 */
xtbloom_status_t update_es2_geometry_cache_cpu(const ES2Plan& plan, const double* positions,
                                               std::uint64_t geometry_generation,
                                               double* matrix_storage,
                                               std::size_t matrix_storage_elements,
                                               const ES2Workspace& workspace,
                                               ES2GeometryCache& cache, std::string& error);

/*
 * Overwrite shell_potentials with v = Gamma*q. Charges are in elementary
 * charge units and potentials are Hartree/e. Packed shell layout follows the
 * plan. workspace.shell_scratch is used for atomic whole-batch publication.
 * Input, output, cache, and active scratch buffers must not overlap. Writable
 * output, cache, and scratch storage must not alias immutable plan storage;
 * active buffers must not alias the plan, cache, or workspace descriptors.
 */
xtbloom_status_t evaluate_es2_potential_cpu(const ES2Plan& plan, const ES2GeometryCache& cache,
                                            const double* shell_charges, double* shell_potentials,
                                            const ES2Workspace& workspace, std::string& error);

/*
 * Overwrite the ES2 shell potential of exactly one ragged batch member. The
 * charge and potential pointers retain the full packed shell layout, but only
 * the selected system's shell slice is read or modified. This lets an SCC
 * worker prepare the Hamiltonian of a successful member while peers may
 * contain NaN or fail independently.
 *
 * Structural and aliasing failures return INVALID_ARGUMENT. Nonfinite
 * target-system data or target arithmetic failure return INTERNAL_ERROR;
 * in that case the target slice of shell_potentials may be partially
 * modified, so callers must treat the whole target system as failed and must
 * not consume its slice. No per-call allocation is performed.
 */
xtbloom_status_t evaluate_es2_potential_system_cpu(const ES2Plan& plan,
                                                   const ES2GeometryCache& cache,
                                                   std::int64_t system, const double* shell_charges,
                                                   double* shell_potentials, std::string& error);

/*
 * Accumulate one Hartree energy per batch member,
 *
 *   E2 = 1/2 q^T Gamma q.
 *
 * The geometry cache is reused unchanged across SCC iterations.
 * workspace.batch_scratch holds one unpublished contribution per member.
 * Writable output, cache, and scratch storage must not alias plan storage;
 * active buffers must not alias the plan, cache, or workspace descriptors.
 */
xtbloom_status_t add_es2_energy_cpu(const ES2Plan& plan, const ES2GeometryCache& cache,
                                    const double* shell_charges, double* energies,
                                    const ES2Workspace& workspace, std::string& error);

/*
 * Accumulate the ES2 energy for exactly one ragged batch member. The charge
 * pointer still addresses the complete packed shell array, but numerical data
 * outside system's shell slice is neither read nor validated. This permits an
 * SCC worker to commit a healthy member even when a peer contains NaN.
 *
 * Structural and binding failures return XTBLOOM_STATUS_INVALID_ARGUMENT.
 * Invalid target-system numerical data or floating-point range failure returns
 * XTBLOOM_STATUS_INTERNAL_ERROR. In either case accumulated_energy is unchanged.
 * The scalar contribution is staged locally, so this one-system primitive
 * requires no caller scratch and performs no allocation.
 */
xtbloom_status_t add_es2_energy_system_cpu(const ES2Plan& plan, const ES2GeometryCache& cache,
                                           std::int64_t system, const double* shell_charges,
                                           double& accumulated_energy, std::string& error);

/*
 * Accumulate the fixed-q coordinate VJP dE2/dR in Hartree/bohr. This routine
 * contracts shell pairs directly into workspace.gradient_scratch and never
 * materializes an atom-pair derivative tensor. geometry_generation must equal
 * cache.geometry_generation: positions are required to be the same geometry
 * generation used to build the cache. Writable output, cache, and scratch
 * storage must not alias plan storage; active buffers must not alias the plan,
 * cache, or workspace descriptors. gradients are derivatives, not forces.
 */
xtbloom_status_t add_es2_gradient_cpu(const ES2Plan& plan, const ES2GeometryCache& cache,
                                      const double* positions, std::uint64_t geometry_generation,
                                      const double* shell_charges, double* gradients,
                                      const ES2Workspace& workspace, std::string& error);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_ES2_HPP
