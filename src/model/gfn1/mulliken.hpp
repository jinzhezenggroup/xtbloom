#ifndef XTBLOOM_MODEL_GFN1_MULLIKEN_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN1_MULLIKEN_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "model/gfn1/basis.hpp"
#include "model/gfn1/integrals.hpp"
#include "model/gfn1/wavefunction.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn1 {

struct MullikenPlanData;

/*
 * Immutable scalar Mulliken topology for GFN1.
 *
 * GFN1 owns only shell charge and, for unrestricted systems, shell
 * magnetization. Keeping this plan distinct from GFN2 prevents accidental
 * allocation or contraction of dipole/quadrupole SCC variables.
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
  [[nodiscard]] std::int64_t population_scratch_elements() const noexcept;
  [[nodiscard]] std::int64_t hamiltonian_scratch_elements() const noexcept;
  [[nodiscard]] std::size_t resident_bytes() const noexcept;
  /* True when a byte range aliases the immutable plan or its backing arrays. */
  [[nodiscard]] bool overlaps_storage(const void* data, std::size_t size_bytes) const noexcept;

  [[nodiscard]] const std::vector<std::int64_t>& atom_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& batch_shell_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& batch_orbital_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& matrix_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& shell_orbital_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& shell_to_atom() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& orbital_to_shell() const noexcept;
  [[nodiscard]] const std::vector<std::int32_t>& spin_channels() const noexcept;
  [[nodiscard]] const std::vector<double>& reference_shell_occupations() const noexcept;
  [[nodiscard]] const MullikenPlanData* identity() const noexcept;

 private:
  explicit MullikenPlan(std::shared_ptr<const MullikenPlanData> data) noexcept;
  std::shared_ptr<const MullikenPlanData> data_;

  friend xtbloom_status_t make_mulliken_plan(const BasisPlan&, const IntegralPlan&,
                                             const WavefunctionLayout&, MullikenPlan&,
                                             std::string&);
};

struct MullikenIntegralView {
  const double* overlap = nullptr;
  std::int64_t elements = 0;
  const MullikenPlanData* plan_identity = nullptr;
};

struct MullikenDensityView {
  const double* density = nullptr;
  std::int64_t elements = 0;
  const MullikenPlanData* plan_identity = nullptr;
};

struct MullikenPopulationView {
  double* qsh = nullptr;
  std::int64_t qsh_elements = 0;
  double* qat = nullptr;
  std::int64_t qat_elements = 0;
  const MullikenPlanData* plan_identity = nullptr;
};

/* Charge/magnetization shell potentials in wavefunction field packing. */
struct MullikenPotentialView {
  const double* vsh = nullptr;
  std::int64_t elements = 0;
  const MullikenPlanData* plan_identity = nullptr;
};

struct MullikenHamiltonianView {
  double* matrix = nullptr;
  std::int64_t elements = 0;
  const MullikenPlanData* plan_identity = nullptr;
};

/* One allocation-free scratch binding, measured in doubles rather than bytes. */
struct MullikenWorkspace {
  double* scratch = nullptr;
  std::int64_t elements = 0;
};

xtbloom_status_t make_mulliken_plan(const BasisPlan& basis, const IntegralPlan& integrals,
                                    const WavefunctionLayout& wavefunction, MullikenPlan& plan,
                                    std::string& error);

/*
 * Compute q_s = n_s^0 - sum_mu P_{mu,nu} S_{mu,nu}, with ket AO nu
 * assigning the contribution to its shell. For two spin channels, alpha/beta
 * contractions become charge and magnetization (alpha-beta), then shell
 * populations are reduced to atoms without introducing a mixed atomic SCC
 * variable.
 */
xtbloom_status_t evaluate_mulliken_population_system_cpu(
    const MullikenPlan& plan, const MullikenIntegralView& integrals,
    const MullikenDensityView& density, const MullikenPopulationView& population,
    std::int64_t system, const MullikenWorkspace& workspace, std::string& error);

xtbloom_status_t evaluate_mulliken_population_cpu(const MullikenPlan& plan,
                                                  const MullikenIntegralView& integrals,
                                                  const MullikenDensityView& density,
                                                  const MullikenPopulationView& population,
                                                  const MullikenWorkspace& workspace,
                                                  std::string& error);

/*
 * Add the scalar SCC shift
 *
 *   dH_mn = -S_mn (v_shell(m) + v_shell(n)) / 2.
 *
 * Charge/magnetization potentials are converted to the effective alpha/beta
 * shifts `v_q +/- v_m`.  Tblite obtains the same matrices by first applying
 * its half-valued magnet-to-up/down transform and then doubling unrestricted
 * Hamiltonians in the eigensolver.  GFN1 keeps H0 unscaled in each explicit
 * spin channel, so the compensating factor belongs in this conversion.
 * The target matrix is unchanged on failure.
 */
xtbloom_status_t add_mulliken_hamiltonian_system_cpu(
    const MullikenPlan& plan, const MullikenIntegralView& integrals,
    const MullikenPotentialView& potential, const MullikenHamiltonianView& hamiltonian,
    std::int64_t system, const MullikenWorkspace& workspace, std::string& error);

xtbloom_status_t add_mulliken_hamiltonian_cpu(const MullikenPlan& plan,
                                              const MullikenIntegralView& integrals,
                                              const MullikenPotentialView& potential,
                                              const MullikenHamiltonianView& hamiltonian,
                                              const MullikenWorkspace& workspace,
                                              std::string& error);

}  // namespace xtbloom::detail::gfn1

#endif  // XTBLOOM_MODEL_GFN1_MULLIKEN_HPP
