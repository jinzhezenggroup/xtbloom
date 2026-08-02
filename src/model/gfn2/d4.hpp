#ifndef GPUXTB_MODEL_GFN2_D4_HPP
#define GPUXTB_MODEL_GFN2_D4_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gpuxtb/gpuxtb.h"

namespace gpuxtb::detail::gfn2 {

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
  [[nodiscard]] const D4PlanData* identity() const noexcept;

 private:
  explicit D4Plan(std::shared_ptr<const D4PlanData> data) noexcept;
  std::shared_ptr<const D4PlanData> data_;

  friend gpuxtb_status_t make_d4_plan(std::int64_t, std::int64_t, const std::int64_t*,
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

gpuxtb_status_t make_d4_plan(std::int64_t batch_size, std::int64_t total_atoms,
                             const std::int64_t* atom_offsets, const std::int32_t* atomic_numbers,
                             D4Plan& plan, std::string& error);

gpuxtb_status_t bind_d4_workspace(const D4Plan& plan, void* workspace, std::size_t workspace_size,
                                  D4Workspace& view, std::string& error);

/*
 * Build D4 CN and pair damping data for atom-major positions in bohr.
 * pair_storage needs 5*total_pairs doubles and coordination_storage needs
 * total_atoms doubles. Publication is atomic and a successful call allocates
 * nothing. Positions, both outputs, the canonical workspace, immutable plan
 * storage, and control descriptors must be mutually disjoint. The same output
 * arrays may be reused to refresh an existing cache generation.
 */
gpuxtb_status_t update_d4_geometry_cache_cpu(
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
gpuxtb_status_t evaluate_d4_two_body_cpu(const D4Plan& plan, const D4GeometryCache& cache,
                                         const double* atomic_charges, double* energies,
                                         double* atomic_potentials, const D4Workspace& workspace,
                                         std::string& error);

/*
 * Accumulate the complete two-body coordinate derivative at fixed atomic
 * charges, including the D4-specific CN interpolation path. Gradients are
 * dE/dR in Hartree/bohr, not forces. The gradient output must not overlap the
 * charges, geometry cache, workspace, plan storage, or descriptors.
 */
gpuxtb_status_t add_d4_two_body_gradient_cpu(const D4Plan& plan, const D4GeometryCache& cache,
                                             const double* atomic_charges, double* gradients,
                                             const D4Workspace& workspace, std::string& error);

/*
 * Overwrite the non-self-consistent q=0 reference-weighted ATM energy. The
 * finite-system implementation uses the GFN2 s9/a1/a2 parameters and the
 * same 25-bohr sharp cutoff as pinned tblite. The output must be disjoint from
 * the geometry cache, workspace, plan storage, and descriptors.
 */
gpuxtb_status_t evaluate_d4_atm_cpu(const D4Plan& plan, const D4GeometryCache& cache,
                                    double* energies, const D4Workspace& workspace,
                                    std::string& error);

/*
 * Accumulate the analytic ATM coordinate derivative, including the CN path.
 * The gradient output obeys the same non-aliasing contract as the two-body
 * gradient operation.
 */
gpuxtb_status_t add_d4_atm_gradient_cpu(const D4Plan& plan, const D4GeometryCache& cache,
                                        double* gradients, const D4Workspace& workspace,
                                        std::string& error);

}  // namespace gpuxtb::detail::gfn2

#endif  // GPUXTB_MODEL_GFN2_D4_HPP
