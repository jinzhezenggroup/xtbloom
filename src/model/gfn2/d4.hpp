#ifndef XTBLOOM_MODEL_GFN2_D4_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_D4_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {

class PeriodicShortRangePlan;
struct PeriodicShortRangeGeometry;
struct PeriodicShortRangeWorkspace;

inline constexpr std::size_t kD4WorkspaceAlignment = 64u;
inline constexpr std::size_t kD4MaximumReferences = 7u;
inline constexpr std::size_t kD4PairDataElements = 5u;

struct D4PlanData;

/*
 * Immutable ragged-batch topology and GFN2 D4 parameters.
 *
 * D4 uses its own electronegativity-weighted error-function coordination
 * number and must not reuse the ordinary GFN2 double-exponential CN. Pair
 * offsets pack one strict lower triangle per system. Setup may allocate;
 * geometry updates and evaluations below use only caller-owned storage.
 */
class D4Plan {
 public:
  D4Plan() noexcept = default;
  D4Plan(const D4Plan&) noexcept = default;
  D4Plan(D4Plan&&) noexcept = default;
  D4Plan& operator=(const D4Plan&) noexcept = default;
  D4Plan& operator=(D4Plan&&) noexcept = default;
  ~D4Plan() = default;

  [[nodiscard]] bool sealed() const noexcept;
  [[nodiscard]] std::int64_t batch_size() const noexcept;
  [[nodiscard]] std::int64_t total_atoms() const noexcept;
  [[nodiscard]] std::int64_t total_pairs() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& atom_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& pair_offsets() const noexcept;
  /* Verify the exact element ordering without exposing mutable plan storage. */
  [[nodiscard]] bool matches_atomic_numbers(const std::int32_t* atomic_numbers) const noexcept;
  /* Reject caller-owned numerical buffers that overlap immutable plan storage. */
  [[nodiscard]] bool overlaps_storage(const void* data, std::size_t size_bytes) const noexcept;
  [[nodiscard]] std::size_t workspace_size_bytes() const noexcept;
  [[nodiscard]] std::size_t resident_bytes() const noexcept;
  [[nodiscard]] const D4PlanData* identity() const noexcept;

 private:
  explicit D4Plan(std::shared_ptr<const D4PlanData> data) noexcept;
  std::shared_ptr<const D4PlanData> data_;

  friend xtbloom_status_t make_d4_plan(std::int64_t, std::int64_t, const std::int64_t*,
                                       const std::int32_t*, D4Plan&, std::string&);
};

/*
 * Reusable geometry cache. Each pair stores
 *
 *   [R_i.x-R_j.x, R_i.y-R_j.y, R_i.z-R_j.z, D_ij, dD_ij/d(r^2)*2]
 *
 * for i<j, where D_ij is the charge-independent rational damping factor
 * multiplying C6. The final scalar contracts directly with R_i-R_j to form
 * the explicit coordinate gradient. Coordination numbers use a separate
 * atom-wise array because they feed the reference interpolation.
 */
struct D4GeometryCache {
  double* pair_data = nullptr;
  std::int64_t pair_data_elements = 0;
  double* coordination_numbers = nullptr;
  std::int64_t coordination_elements = 0;
  std::uint64_t geometry_generation = 0u;
  const D4PlanData* plan_identity = nullptr;
};

/*
 * Caller-owned unpublished storage; all counts are numbers of doubles.
 * bind_d4_workspace creates a canonical pointer layout. Callers may retain
 * the descriptor but must not modify its nested pointers, extents, or plan
 * identity before an update/evaluation call.
 */
struct D4Workspace {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0u;
  double* pair_scratch = nullptr;
  std::int64_t pair_elements = 0;
  double* coordination_scratch = nullptr;
  std::int64_t coordination_elements = 0;
  double* weights = nullptr;
  double* weight_cn_derivatives = nullptr;
  double* weight_charge_derivatives = nullptr;
  std::int64_t weight_elements = 0;
  double* atom_scratch = nullptr;
  double* coordination_adjoints = nullptr;
  std::int64_t atom_elements = 0;
  double* batch_scratch = nullptr;
  std::int64_t batch_elements = 0;
  double* gradient_scratch = nullptr;
  std::int64_t gradient_elements = 0;
  const D4PlanData* plan_identity = nullptr;
};

xtbloom_status_t make_d4_plan(std::int64_t batch_size, std::int64_t total_atoms,
                              const std::int64_t* atom_offsets, const std::int32_t* atomic_numbers,
                              D4Plan& plan, std::string& error);

xtbloom_status_t bind_d4_workspace(const D4Plan& plan, void* workspace, std::size_t workspace_size,
                                   D4Workspace& view, std::string& error);

/*
 * Build D4 CN and pair damping data for atom-major positions in bohr.
 * pair_storage needs 5*total_pairs doubles and coordination_storage needs
 * total_atoms doubles. Publication is atomic and a successful call allocates
 * nothing. Positions, both outputs, the canonical workspace, immutable plan
 * storage, and control descriptors must be mutually disjoint. The same output
 * arrays may be reused to refresh an existing cache generation.
 */
xtbloom_status_t update_d4_geometry_cache_cpu(
    const D4Plan& plan, const double* positions, std::uint64_t geometry_generation,
    double* pair_storage, std::size_t pair_storage_elements, double* coordination_storage,
    std::size_t coordination_storage_elements, const D4Workspace& workspace, D4GeometryCache& cache,
    std::string& error);

/*
 * Overwrite the per-system self-consistent two-body energy and atom potential
 * dE_D4/dq. Charges are GFN2 Mulliken atomic charges. This is the SCC-facing
 * operation; the fixed-geometry damping cache is reused at every iteration.
 * Active inputs, outputs, workspace, plan storage, and descriptors must not
 * overlap.
 */
xtbloom_status_t evaluate_d4_two_body_cpu(const D4Plan& plan, const D4GeometryCache& cache,
                                          const double* atomic_charges, double* energies,
                                          double* atomic_potentials, const D4Workspace& workspace,
                                          std::string& error);

/*
 * Evaluate one ragged batch member using full-layout atomic charges. Only the
 * selected charge, coordination, and pair slices are inspected, and only the
 * selected atomic-potential slice is published. Passing nullptr for
 * atomic_potentials selects energy-only mode, which is used to recompute the
 * final D4 energy from raw post-SCC charges without disturbing mixed-charge
 * SCC potentials.
 *
 * Structural and aliasing failures return INVALID_ARGUMENT. Invalid target
 * numerical data or arithmetic failure return INTERNAL_ERROR. Energy and the
 * optional target potential slice remain unchanged on every failure. The
 * operation uses canonical caller-owned scratch and allocates nothing.
 */
xtbloom_status_t evaluate_d4_two_body_system_cpu(const D4Plan& plan, const D4GeometryCache& cache,
                                                 std::int64_t system, const double* atomic_charges,
                                                 double& energy, double* atomic_potentials,
                                                 const D4Workspace& workspace, std::string& error);

/*
 * Accumulate the complete two-body coordinate derivative at fixed atomic
 * charges, including the D4-specific CN interpolation path. Gradients are
 * dE/dR in Hartree/bohr, not forces. The gradient output must not overlap the
 * charges, geometry cache, workspace, plan storage, or descriptors.
 */
xtbloom_status_t add_d4_two_body_gradient_cpu(const D4Plan& plan, const D4GeometryCache& cache,
                                              const double* atomic_charges, double* gradients,
                                              const D4Workspace& workspace, std::string& error);

/*
 * Overwrite the non-self-consistent q=0 reference-weighted ATM energy. The
 * molecular finite-system implementation uses the GFN2 s9/a1/a2 parameters
 * and the existing 25-bohr sharp cutoff as pinned tblite. The periodic ATM
 * path below uses its separately documented smooth outer switch. The output
 * must be disjoint from the geometry cache, workspace, plan storage, and
 * descriptors.
 */
xtbloom_status_t evaluate_d4_atm_cpu(const D4Plan& plan, const D4GeometryCache& cache,
                                     double* energies, const D4Workspace& workspace,
                                     std::string& error);

/*
 * Accumulate the analytic ATM coordinate derivative, including the CN path.
 * The gradient output obeys the same non-aliasing contract as the two-body
 * gradient operation.
 */
xtbloom_status_t add_d4_atm_gradient_cpu(const D4Plan& plan, const D4GeometryCache& cache,
                                         double* gradients, const D4Workspace& workspace,
                                         std::string& error);

/*
 * Periodic D4 uses the complete real-space image lists owned by the shared
 * short-range plan.  The D4 plan still owns all element/reference data, while
 * the prepared periodic geometry supplies wrapped central-cell coordinates.
 * D4 coordination retains its sharp 30-bohr cutoff. The periodic two-body
 * and ATM terms use the pinned tblite 0.05-bohr (in bohr) quintic switch at
 * their 50- and 25-bohr outer cutoffs, including the switch derivative in
 * Cartesian and affine-cell adjoints. ATM applies the product of the three
 * pair switches and the complete product-rule derivative. The output of each
 * evaluate_* operation is overwritten only after its complete image sum
 * succeeds; add_* operations accumulate only after their complete derivative
 * sum succeeds. Workspace scratch may be changed on failure, but active
 * inputs, outputs, workspaces, plans, and descriptors must be disjoint.
 */
xtbloom_status_t evaluate_periodic_d4_coordination_cpu(const D4Plan& plan,
                                                       const PeriodicShortRangePlan& periodic_plan,
                                                       const PeriodicShortRangeGeometry& geometry,
                                                       double* coordination_numbers,
                                                       const PeriodicShortRangeWorkspace& workspace,
                                                       std::string& error);

/* Accumulate the D4-CN VJP into dE/dR and row-major dE/d epsilon. */
xtbloom_status_t add_periodic_d4_coordination_gradient_cpu(
    const D4Plan& plan, const PeriodicShortRangePlan& periodic_plan,
    const PeriodicShortRangeGeometry& geometry, const double* dE_dcn, double* gradients,
    double* strain_derivatives, const PeriodicShortRangeWorkspace& workspace, std::string& error);

/*
 * Evaluate the charge-dependent periodic two-body term.  Per-atom energies
 * use the tblite lower-triangle accounting (one half to each endpoint and a
 * half-weighted self image). For first == second, the zero translation is
 * excluded. Each nonzero self-image pair is half-weighted because opposite
 * directed self images would otherwise be counted twice; its Cartesian
 * derivative cancels because both image endpoints follow the same central
 * atom coordinate, while its affine strain derivative remains. Potentials
 * are dE/dq in Hartree per electron.
 */
xtbloom_status_t evaluate_periodic_d4_two_body_cpu(
    const D4Plan& plan, const PeriodicShortRangePlan& periodic_plan,
    const PeriodicShortRangeGeometry& geometry, const double* coordination_numbers,
    const double* atomic_charges, double* per_atom_energies, double* atomic_potentials,
    const D4Workspace& d4_workspace, const PeriodicShortRangeWorkspace& workspace,
    std::string& error);

/* Add periodic two-body dE/dR and dE/d epsilon, including the D4-CN path. */
xtbloom_status_t add_periodic_d4_two_body_gradient_cpu(
    const D4Plan& plan, const PeriodicShortRangePlan& periodic_plan,
    const PeriodicShortRangeGeometry& geometry, const double* coordination_numbers,
    const double* atomic_charges, double* gradients, double* strain_derivatives,
    const D4Workspace& d4_workspace, const PeriodicShortRangeWorkspace& workspace,
    std::string& error);

/*
 * Evaluate the zero-charge/non-SCC periodic ATM term into per-atom energies.
 * The lower-triangular central-atom loops use multiplicities 1, 1/2, and 1/6
 * for three distinct, two equal, and three equal central-atom labels. This
 * handles permutation multiplicity only: lattice translations remain fully
 * enumerated and receive no Wigner--Seitz 1/n weight. Zero self translations
 * are excluded on repeated legs, and the validated coincident image rule is
 * applied to the remaining edges.
 */
xtbloom_status_t evaluate_periodic_d4_atm_cpu(
    const D4Plan& plan, const PeriodicShortRangePlan& periodic_plan,
    const PeriodicShortRangeGeometry& geometry, const double* coordination_numbers,
    double* per_atom_energies, const D4Workspace& d4_workspace,
    const PeriodicShortRangeWorkspace& workspace, std::string& error);

/* Add periodic ATM dE/dR and dE/d epsilon, including its D4-CN path and the
 * product-rule derivative of the three pair cutoff switches. */
xtbloom_status_t add_periodic_d4_atm_gradient_cpu(
    const D4Plan& plan, const PeriodicShortRangePlan& periodic_plan,
    const PeriodicShortRangeGeometry& geometry, const double* coordination_numbers,
    double* gradients, double* strain_derivatives, const D4Workspace& d4_workspace,
    const PeriodicShortRangeWorkspace& workspace, std::string& error);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_D4_HPP
