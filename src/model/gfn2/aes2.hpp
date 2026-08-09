#ifndef XTBLOOM_MODEL_GFN2_AES2_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_AES2_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "model/gfn2/basis.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {

struct AES2PlanData;

/*
 * Immutable atom topology and element parameters for GFN2 anisotropic
 * second-order electrostatics (AES2). Copying a plan is O(1). Pair offsets
 * describe the strict lower triangle of every ragged batch member, enumerated
 * as (first, second) with atom_begin <= first < second < atom_end.
 *
 * Multipoles use atom-major layouts: dipoles have three contiguous Cartesian
 * components and quadrupoles have six packed components in
 * [xx,xy,yy,xz,yz,zz] order. Quadrupoles supplied to the numerical routines
 * are expected to be the traceless cumulative atomic moments produced by the
 * GFN2 Mulliken analysis.
 */
class AES2Plan {
 public:
  AES2Plan() noexcept = default;
  AES2Plan(const AES2Plan&) noexcept = default;
  AES2Plan(AES2Plan&&) noexcept = default;
  AES2Plan& operator=(const AES2Plan&) noexcept = default;
  AES2Plan& operator=(AES2Plan&&) noexcept = default;
  ~AES2Plan() = default;

  [[nodiscard]] bool sealed() const noexcept;
  [[nodiscard]] std::int64_t batch_size() const noexcept;
  [[nodiscard]] std::int64_t total_atoms() const noexcept;
  [[nodiscard]] std::int64_t total_pairs() const noexcept;
  [[nodiscard]] std::int64_t pair_data_elements() const noexcept;
  [[nodiscard]] std::int64_t potential_scratch_elements() const noexcept;
  [[nodiscard]] std::int64_t gradient_scratch_elements() const noexcept;
  [[nodiscard]] std::int64_t coordination_scratch_elements() const noexcept;
  [[nodiscard]] std::size_t resident_bytes() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& atom_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& pair_offsets() const noexcept;
  [[nodiscard]] const std::vector<double>& dipole_kernel() const noexcept;
  [[nodiscard]] const std::vector<double>& quadrupole_kernel() const noexcept;
  [[nodiscard]] const std::vector<double>& multipole_radius() const noexcept;
  [[nodiscard]] const std::vector<double>& multipole_valence_cn() const noexcept;

  /* True when a byte range aliases this plan's immutable object or backing storage. */
  [[nodiscard]] bool overlaps_storage(const void* data, std::size_t size_bytes) const noexcept;

  /* Stable compatibility token for backend caches and diagnostics. */
  [[nodiscard]] const AES2PlanData* identity() const noexcept;

 private:
  explicit AES2Plan(std::shared_ptr<const AES2PlanData> data) noexcept;

  std::shared_ptr<const AES2PlanData> data_;

  friend xtbloom_status_t make_aes2_plan(const BasisPlan& basis, const std::int32_t* atomic_numbers,
                                         AES2Plan& plan, std::string& error);
};

/*
 * Non-owning compact geometry cache. Each unordered atom pair stores exactly
 * five doubles in AoS order:
 *
 *   [R_first.x-R_second.x, R_first.y-R_second.y,
 *    R_first.z-R_second.z, R^-3*f3, R^-5*f5].
 *
 * This is sufficient to reconstruct q-d, d-d, and q-Q kernels during every
 * SCC iteration while using substantially less memory than three dense
 * interaction tensors. The same scalars are also the natural inputs to a
 * future coordinate/CN reverse-mode kernel.
 */
struct AES2GeometryCache {
  double* pair_data = nullptr;
  std::int64_t pair_data_elements = 0;
  std::uint64_t geometry_generation = 0;
  const AES2PlanData* plan_identity = nullptr;
};

/*
 * Caller-owned unpublished storage used for whole-batch failure atomicity.
 * Counts are doubles, not bytes. Cache update requires pair_data_elements;
 * potential evaluation requires 10*total_atoms values; energy accumulation
 * requires batch_size values; the coordinate/CN VJP requires 3*total_atoms
 * gradient values and total_atoms coordination values. Successful steady-state
 * calls allocate nothing.
 */
struct AES2Workspace {
  double* pair_scratch = nullptr;
  std::int64_t pair_elements = 0;
  double* potential_scratch = nullptr;
  std::int64_t potential_elements = 0;
  double* batch_scratch = nullptr;
  std::int64_t batch_elements = 0;
  double* gradient_scratch = nullptr;
  std::int64_t gradient_elements = 0;
  double* coordination_scratch = nullptr;
  std::int64_t coordination_elements = 0;
};

/*
 * Build a GFN2 AES2 plan from an exact BasisPlan/element ordering. The basis
 * shell metadata is cross-checked against the generated parameter table so a
 * same-sized but mismatched atomic-number list is rejected.
 */
xtbloom_status_t make_aes2_plan(const BasisPlan& basis, const std::int32_t* atomic_numbers,
                                AES2Plan& plan, std::string& error);

/*
 * Build the compact pair cache for positions in bohr and externally evaluated
 * GFN2 coordination numbers. The CN-dependent multipole radius is
 *
 *   mrad_A = rad_A + (rmax-rad_A) /
 *                       (1 + exp[-kexp*(CN_A-vCN_A-shift)]),
 *
 * and the finite-system damping functions are
 *
 *   f_n = 1 / (1 + 6*[(mrad_A+mrad_B)/(2*R_AB)]^dmp_n),
 *
 * with dmp3=3 and dmp5=4 for GFN2. pair_storage and
 * workspace.pair_scratch must each hold plan.pair_data_elements() doubles and
 * must be disjoint. On failure, pair_storage and cache are unchanged.
 */
xtbloom_status_t update_aes2_geometry_cache_cpu(
    const AES2Plan& plan, const double* positions, const double* coordination_numbers,
    std::uint64_t geometry_generation, double* pair_storage, std::size_t pair_storage_elements,
    const AES2Workspace& workspace, AES2GeometryCache& cache, std::string& error);

/*
 * Overwrite atomic q/d/Q potentials, i.e. the derivatives of AES2 energy with
 * respect to the corresponding packed multipoles. For pair vector r=R_i-R_j,
 *
 *   A_sd = r R^-3 f3,
 *   A_dd = I R^-3 f5 - 3 r r^T R^-5 f5,
 *   A_sq = [x^2,2xy,y^2,2xz,2yz,z^2] R^-5 f5.
 *
 * Onsite terms are dkernel*|d|^2 and qkernel*(Qxx^2+2Qxy^2+
 * Qyy^2+2Qxz^2+2Qyz^2+Qzz^2). All inputs, outputs, active cache storage,
 * and workspace.potential_scratch must be mutually disjoint. No output is
 * modified if validation or arithmetic fails.
 */
xtbloom_status_t evaluate_aes2_potential_cpu(const AES2Plan& plan, const AES2GeometryCache& cache,
                                             const double* atomic_charges,
                                             const double* atomic_dipoles,
                                             const double* atomic_quadrupoles,
                                             double* charge_potentials, double* dipole_potentials,
                                             double* quadrupole_potentials,
                                             const AES2Workspace& workspace, std::string& error);

/*
 * Overwrite the AES2 atom q/d/Q potentials of exactly one ragged batch member.
 * The multipole and potential pointers retain the full atom-major layout, but
 * only the selected system's atom and pair slices are inspected or written.
 * This lets an SCC worker prepare the Hamiltonian for a successful member
 * while peers may fail independently.
 *
 * Structural and aliasing failures return INVALID_ARGUMENT. Nonfinite target
 * data or target arithmetic failure return INTERNAL_ERROR and the target
 * potential slices remain unchanged. The canonical caller-owned scratch is
 * used for staging, and successful calls allocate nothing.
 */
xtbloom_status_t evaluate_aes2_potential_system_cpu(
    const AES2Plan& plan, const AES2GeometryCache& cache, std::int64_t system,
    const double* atomic_charges, const double* atomic_dipoles, const double* atomic_quadrupoles,
    double* charge_potentials, double* dipole_potentials, double* quadrupole_potentials,
    const AES2Workspace& workspace, std::string& error);

/*
 * Accumulate one AES2 energy per ragged batch member. Existing energies must
 * be finite. workspace.batch_scratch stages every contribution before any
 * caller output is updated, preserving call-level failure atomicity.
 */
xtbloom_status_t add_aes2_energy_cpu(const AES2Plan& plan, const AES2GeometryCache& cache,
                                     const double* atomic_charges, const double* atomic_dipoles,
                                     const double* atomic_quadrupoles, double* energies,
                                     const AES2Workspace& workspace, std::string& error);

/*
 * Accumulate the AES2 energy of one ragged batch member into
 * accumulated_energy. The multipole pointers retain the full-batch layout,
 * but only the selected atom and pair slices are inspected. Consequently a
 * numerical failure in another system cannot poison a successful target.
 *
 * Structural and aliasing failures return INVALID_ARGUMENT. Nonfinite target
 * data or target arithmetic failure return INTERNAL_ERROR. Publication is
 * atomic: accumulated_energy is unchanged unless the complete target
 * contribution is finite. The canonical batch scratch remains caller-owned,
 * and successful calls allocate nothing.
 */
xtbloom_status_t add_aes2_energy_system_cpu(const AES2Plan& plan, const AES2GeometryCache& cache,
                                            std::int64_t system, const double* atomic_charges,
                                            const double* atomic_dipoles,
                                            const double* atomic_quadrupoles,
                                            double& accumulated_energy,
                                            const AES2Workspace& workspace, std::string& error);

/*
 * Accumulate the fixed-multipole reverse products dE_AES2/dR (Hartree/bohr)
 * and dE_AES2/dCN into caller-owned outputs. The Cartesian derivative is the
 * explicit pair-kernel derivative at fixed coordination numbers; callers can
 * subsequently contract coordination_adjoints through the coordination model
 * with add_coordination_gradient_cpu. gradients are derivatives, not forces.
 *
 * The compact cache supplies every pair displacement and damped inverse-power
 * kernel. coordination_numbers are needed only for the derivative of
 *
 *   mrad_A = rad_A + (rmax-rad_A) logistic(kexp*(CN_A-vCN_A-shift)).
 *
 * geometry_generation must match cache.geometry_generation, thereby binding
 * the supplied CN values to the cached geometry. GFN2 dipole/quadrupole onsite
 * kernels are element constants in tblite, xtb, and dxtb; at fixed multipoles
 * their explicit coordinate and CN VJPs are therefore exactly zero.
 * For each r=R_i-R_j pair, the differentiated energy is
 *
 *   E_ij = k3*r.(q_i*d_j-q_j*d_i)
 *        + k5*[R^2*d_i.d_j-3(r.d_i)(r.d_j)+r^T*T*r],
 *   T = q_i*Q_j + q_j*Q_i,
 *
 * where k3=R^-3*f3, k5=R^-5*f5 and rho=(mrad_i+mrad_j)/2.
 * Thus dE/dR_i receives +dE/dr, dE/dR_j receives -dE/dr, and each
 * per-atom radius path receives one half of dE/drho before dmrad/dCN.
 *
 * workspace.gradient_scratch and workspace.coordination_scratch stage both
 * complete outputs so failure is atomic. All active inputs, outputs, cache
 * storage, and scratch buffers must be mutually disjoint and must not overlap
 * plan or descriptor storage. Successful calls allocate nothing.
 */
xtbloom_status_t add_aes2_vjp_cpu(const AES2Plan& plan, const AES2GeometryCache& cache,
                                  const double* positions, const double* coordination_numbers,
                                  std::uint64_t geometry_generation, const double* atomic_charges,
                                  const double* atomic_dipoles, const double* atomic_quadrupoles,
                                  double* gradients, double* coordination_adjoints,
                                  const AES2Workspace& workspace, std::string& error);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_AES2_HPP
