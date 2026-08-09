#ifndef GPUXTB_MODEL_GFN2_MULLIKEN_HPP
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_MODEL_GFN2_MULLIKEN_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gpuxtb/gpuxtb.h"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/integrals.hpp"
#include "model/gfn2/parallel_executor.hpp"
#include "model/gfn2/wavefunction.hpp"

namespace gpuxtb::detail::gfn2 {

struct MullikenPlanData;

/*
 * Immutable, backend-neutral topology for the density-to-multipole and
 * potential-to-H maps used in every GFN2 SCC iteration.
 *
 * Construction validates and seals the exact ragged BasisPlan,
 * IntegralPlan, and WavefunctionLayout relationship. Numerical calls then
 * use only flat POD views, making the same offsets suitable for CPU, CUDA,
 * and a future ROCm backend. Copying a plan is O(1).
 */
class MullikenPlan {
 public:
  MullikenPlan() noexcept = default;
  MullikenPlan(const MullikenPlan&) noexcept = default;
  MullikenPlan(MullikenPlan&&) noexcept = default;
  MullikenPlan& operator=(const MullikenPlan&) noexcept = default;
  MullikenPlan& operator=(MullikenPlan&&) noexcept = default;
  ~MullikenPlan() = default;

  [[nodiscard]] bool sealed() const noexcept;
  [[nodiscard]] std::int64_t batch_size() const noexcept;
  [[nodiscard]] std::int64_t total_atoms() const noexcept;
  [[nodiscard]] std::int64_t total_shells() const noexcept;
  [[nodiscard]] std::int64_t total_orbitals() const noexcept;
  [[nodiscard]] std::int64_t matrix_elements() const noexcept;
  [[nodiscard]] std::int64_t density_elements() const noexcept;
  [[nodiscard]] std::int64_t shell_population_elements() const noexcept;
  [[nodiscard]] std::int64_t atom_population_elements() const noexcept;
  [[nodiscard]] std::int64_t dipole_population_elements() const noexcept;
  [[nodiscard]] std::int64_t quadrupole_population_elements() const noexcept;
  [[nodiscard]] std::int64_t population_scratch_elements() const noexcept;
  [[nodiscard]] std::int64_t hamiltonian_scratch_elements() const noexcept;
  [[nodiscard]] std::size_t resident_bytes() const noexcept;

  [[nodiscard]] const std::vector<std::int64_t>& atom_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& batch_shell_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& batch_orbital_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& matrix_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& shell_orbital_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& shell_to_atom() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& orbital_to_shell() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& orbital_to_atom() const noexcept;
  [[nodiscard]] const std::vector<std::int32_t>& spin_channels() const noexcept;
  [[nodiscard]] const std::vector<double>& reference_shell_occupations() const noexcept;

  /* True when a byte range aliases this plan's immutable object or backing storage. */
  [[nodiscard]] bool overlaps_storage(const void* data, std::size_t size_bytes) const noexcept;

  /* Opaque stable token for backend cache compatibility and diagnostics. */
  [[nodiscard]] const MullikenPlanData* identity() const noexcept;

 private:
  explicit MullikenPlan(std::shared_ptr<const MullikenPlanData> data) noexcept;

  std::shared_ptr<const MullikenPlanData> data_;

  friend gpuxtb_status_t make_mulliken_plan(const BasisPlan& basis, const IntegralPlan& integrals,
                                            const WavefunctionLayout& wavefunction,
                                            MullikenPlan& plan, std::string& error);
};

/*
 * Integral matrices use IntegralPlan's global component-major packing:
 * overlap has M values, dipole has 3*M, and quadrupole has 6*M in
 * [xx,xy,yy,xz,yz,zz] order. Rows are bra AOs and columns are ket AOs;
 * dipole and quadrupole origins belong to the ket AO's atom. plan_identity
 * must be the exact token returned by the MullikenPlan used for evaluation.
 */
struct MullikenIntegralView {
  const double* overlap = nullptr;
  const double* dipole = nullptr;
  const double* quadrupole = nullptr;
  std::int64_t matrix_elements = 0;
  const MullikenPlanData* plan_identity = nullptr;
};

/*
 * Density matrices are ragged system-major, then spin-major row-major.
 * plan_identity must match the exact evaluation plan.
 */
struct MullikenDensityView {
  const double* density = nullptr;
  std::int64_t elements = 0;
  const MullikenPlanData* plan_identity = nullptr;
};

/*
 * SCC population outputs use WavefunctionLayout-compatible packing. qsh and
 * qat are system-major then charge/magnetization-channel-major. Atomic
 * multipoles are system-major [channel][atom][component], with components
 * contiguous and quadrupoles ordered [xx,xy,yy,xz,yz,zz]. plan_identity must
 * match the exact evaluation plan.
 */
struct MullikenPopulationView {
  double* qsh = nullptr;
  std::int64_t qsh_elements = 0;
  double* qat = nullptr;
  std::int64_t qat_elements = 0;
  double* dipole = nullptr;
  std::int64_t dipole_elements = 0;
  double* quadrupole = nullptr;
  std::int64_t quadrupole_elements = 0;
  const MullikenPlanData* plan_identity = nullptr;
};

/*
 * Collected SCC potentials in charge/magnetization representation. Scalar AO
 * potentials are formed as vat[AO atom] + vsh[AO shell]. Atomic multipoles
 * use the same [channel][atom][component] packing as MullikenPopulationView.
 * plan_identity must match the exact evaluation plan.
 */
struct MullikenPotentialView {
  const double* vat = nullptr;
  std::int64_t vat_elements = 0;
  const double* vsh = nullptr;
  std::int64_t vsh_elements = 0;
  const double* dipole = nullptr;
  std::int64_t dipole_elements = 0;
  const double* quadrupole = nullptr;
  std::int64_t quadrupole_elements = 0;
  const MullikenPlanData* plan_identity = nullptr;
};

/*
 * Hamiltonians use the same ragged system/spin/matrix packing as density.
 * plan_identity must match the exact evaluation plan.
 */
struct MullikenHamiltonianView {
  double* matrix = nullptr;
  std::int64_t elements = 0;
  const MullikenPlanData* plan_identity = nullptr;
};

/*
 * Caller-owned unpublished output storage used for whole-batch failure
 * atomicity. population evaluation requires population_scratch_elements;
 * Hamiltonian assembly requires hamiltonian_scratch_elements. Counts are
 * doubles, not bytes, and the same allocation may be reused sequentially.
 * Every active numerical range must be disjoint from all other active ranges,
 * the plan and its immutable backing storage, and the view/workspace descriptor
 * objects themselves. Exact and partial overlap are both rejected before any
 * numerical buffer is dereferenced.
 */
struct MullikenWorkspace {
  double* scratch = nullptr;
  std::int64_t elements = 0;
};

gpuxtb_status_t make_mulliken_plan(const BasisPlan& basis, const IntegralPlan& integrals,
                                   const WavefunctionLayout& wavefunction, MullikenPlan& plan,
                                   std::string& error);

/*
 * Compute q = n0 - P:S and atomic d/Q = -P:D/Q for exactly one ragged batch
 * member. The density and population pointers retain the full batch layout,
 * but only the selected system's density slice is read and only its
 * population slices are written. This lets an SCC worker publish a successful
 * member while peers may fail independently.
 *
 * Structural and aliasing failures return INVALID_ARGUMENT. Invalid target
 * numerical data or target arithmetic failure return INTERNAL_ERROR and the
 * target population slices remain unchanged. The canonical caller-owned
 * workspace is used for staging, and successful calls allocate nothing.
 */
gpuxtb_status_t evaluate_mulliken_population_system_cpu(
    const MullikenPlan& plan, const MullikenIntegralView& integrals,
    const MullikenDensityView& density, const MullikenPopulationView& population,
    std::int64_t system, const MullikenWorkspace& workspace, std::string& error,
    const SccParallelExecutor* parallel = nullptr);

/*
 * Compute q = n0 - P:S and atomic d/Q = -P:D/Q. The ket AO determines the
 * owning shell and atom. For two spin channels, alpha/beta contractions are
 * converted to charge and magnetization, with magnetization N_beta-N_alpha.
 * All outputs are overwritten atomically; successful calls allocate nothing.
 */
gpuxtb_status_t evaluate_mulliken_population_cpu(const MullikenPlan& plan,
                                                 const MullikenIntegralView& integrals,
                                                 const MullikenDensityView& density,
                                                 const MullikenPopulationView& population,
                                                 const MullikenWorkspace& workspace,
                                                 std::string& error);

/*
 * Accumulate the scalar, dipole, and quadrupole SCC potential shifts of
 * exactly one ragged batch member into its Hamiltonian slice:
 *
 *   dH_mn = -S_mn (u_m + u_n)/2
 *            -D_mn(A_n):v_D(A_n)/2 - D_nm(A_m):v_D(A_m)/2
 *            -Q_mn(A_n):v_Q(A_n)/2 - Q_nm(A_m):v_Q(A_m)/2.
 *
 * Q uses the packed dual contraction with no extra factor for off-diagonal
 * components. Charge/magnetization potentials are converted to alpha/beta
 * before assembly. The potential and Hamiltonian pointers retain the full
 * batch layout, but only the target system's slices are read or modified.
 * This lets an SCC worker prepare the Hamiltonian for a successful member
 * while peers may fail independently.
 *
 * Structural and aliasing failures return INVALID_ARGUMENT. Invalid target
 * numerical data or target arithmetic failure return INTERNAL_ERROR and the
 * target Hamiltonian slice remains unchanged. The canonical caller-owned
 * workspace is used for staging, and successful calls allocate nothing.
 */
gpuxtb_status_t add_mulliken_hamiltonian_system_cpu(
    const MullikenPlan& plan, const MullikenIntegralView& integrals,
    const MullikenPotentialView& potential, const MullikenHamiltonianView& hamiltonian,
    std::int64_t system, const MullikenWorkspace& workspace, std::string& error,
    const SccParallelExecutor* parallel = nullptr);

/*
 * Accumulate scalar, dipole, and quadrupole SCC potential shifts into H:
 *
 *   dH_mn = -S_mn (u_m + u_n)/2
 *            -D_mn(A_n):v_D(A_n)/2 - D_nm(A_m):v_D(A_m)/2
 *            -Q_mn(A_n):v_Q(A_n)/2 - Q_nm(A_m):v_Q(A_m)/2.
 *
 * Q uses the packed dual contraction with no extra factor for off-diagonal
 * components. Charge/magnetization potentials are converted to alpha/beta
 * before assembly. H is unchanged on any failure.
 */
gpuxtb_status_t add_mulliken_hamiltonian_cpu(const MullikenPlan& plan,
                                             const MullikenIntegralView& integrals,
                                             const MullikenPotentialView& potential,
                                             const MullikenHamiltonianView& hamiltonian,
                                             const MullikenWorkspace& workspace,
                                             std::string& error);

}  // namespace gpuxtb::detail::gfn2

#endif  // GPUXTB_MODEL_GFN2_MULLIKEN_HPP
