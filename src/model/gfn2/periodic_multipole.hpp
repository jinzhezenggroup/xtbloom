// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#ifndef XTBLOOM_MODEL_GFN2_PERIODIC_MULTIPOLE_HPP
#define XTBLOOM_MODEL_GFN2_PERIODIC_MULTIPOLE_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "model/gfn2/aes2.hpp"
#include "model/gfn2/periodic_topology.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {

struct PeriodicMultipolePlanData;

/*
 * Immutable geometry-independent state for the periodic GFN2 AES2 primitive.
 *
 * The public execution path eventually composes this evaluator with the SCC
 * driver.  Keeping the evaluator independent is intentional: the matrix,
 * potential, fixed-multipole derivative, and damping-radius adjoint can be
 * checked against the pinned tblite term corpus before any density iteration
 * or backend-specific staging is involved.
 */
class PeriodicMultipolePlan {
 public:
  PeriodicMultipolePlan() noexcept = default;
  PeriodicMultipolePlan(const PeriodicMultipolePlan&) noexcept = default;
  PeriodicMultipolePlan(PeriodicMultipolePlan&&) noexcept = default;
  PeriodicMultipolePlan& operator=(const PeriodicMultipolePlan&) noexcept = default;
  PeriodicMultipolePlan& operator=(PeriodicMultipolePlan&&) noexcept = default;
  ~PeriodicMultipolePlan() = default;

  [[nodiscard]] bool sealed() const noexcept { return data_ != nullptr; }
  [[nodiscard]] std::int64_t batch_size() const noexcept;
  [[nodiscard]] std::int64_t total_atoms() const noexcept;
  [[nodiscard]] std::int64_t matrix_elements() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& atom_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& matrix_offsets() const noexcept;
  [[nodiscard]] const Lattice3D& lattice(std::int64_t system) const noexcept;
  [[nodiscard]] double alpha(std::int64_t system) const noexcept;
  [[nodiscard]] const std::vector<double>& dipole_kernel() const noexcept;
  [[nodiscard]] const std::vector<double>& quadrupole_kernel() const noexcept;
  [[nodiscard]] const std::vector<double>& multipole_radius() const noexcept;
  [[nodiscard]] const std::vector<double>& multipole_valence_cn() const noexcept;
  [[nodiscard]] const PeriodicMultipolePlanData* identity() const noexcept { return data_.get(); }
  [[nodiscard]] bool overlaps_storage(const void* data, std::size_t bytes) const noexcept;

 private:
  explicit PeriodicMultipolePlan(std::shared_ptr<const PeriodicMultipolePlanData> data) noexcept
      : data_(std::move(data)) {}

  std::shared_ptr<const PeriodicMultipolePlanData> data_;

  friend xtbloom_status_t make_periodic_multipole_plan(const AES2Plan&,
                                                       const PeriodicShortRangePlan&,
                                                       PeriodicMultipolePlan&, std::string&);
  friend xtbloom_status_t evaluate_periodic_multipole_cpu(const PeriodicMultipolePlan&,
                                                          const double*, const double*,
                                                          const double*, const double*,
                                                          const double*, double*, double*, double*,
                                                          double*, double*, double*, double*,
                                                          double*, double*, double*, std::string&);
};

/*
 * Build the periodic multipole plan from the same AES2 parameters and cell
 * topology used by the finite-system GFN2 path.  Direct images use tblite's
 * reviewed 100-bohr multipole radius; reciprocal images use the independent
 * multipole cutoff search and the multipole-specific alpha selection.
 */
xtbloom_status_t make_periodic_multipole_plan(const AES2Plan& aes2,
                                              const PeriodicShortRangePlan& topology,
                                              PeriodicMultipolePlan& plan, std::string& error);

/*
 * Evaluate periodic q/d/Q AES2 for a fixed set of atom multipoles.
 *
 * Input layouts are atom-major: charges[n], dipoles[3*n], and quadrupoles
 * [6*n] with quadrupole order [xx,xy,yy,xz,yz,zz].  The three interaction
 * matrices are packed per local atom pair in each ragged system:
 *
 *   sd[3*(row*n+column)+component],
 *   dd[9*(row*n+column)+3*row_component+column_component],
 *   sq[6*(row*n+column)+component].
 *
 * Potentials are atom-major and include the onsite AXC kernels.  Energies
 * include AES plus onsite AXC.  gradients and strain_derivatives contain the
 * explicit fixed-damping-radius derivatives only; coordination_adjoint, when
 * non-NULL, receives dE/dCN after applying d(mrad)/dCN.  The caller can pass
 * that adjoint to add_periodic_coordination_gradient_cpu to complete the
 * geometry derivative without materializing an O(n^2) CN Jacobian.
 *
 * Strain is row-major dE/d(epsilon), with positions and the direct lattice
 * deformed together.  All outputs are transactional: no output is changed
 * if validation or arithmetic fails.
 */
xtbloom_status_t evaluate_periodic_multipole_cpu(
    const PeriodicMultipolePlan& plan, const double* positions, const double* coordination_numbers,
    const double* atomic_charges, const double* atomic_dipoles, const double* atomic_quadrupoles,
    double* charge_dipole_matrix, double* dipole_dipole_matrix, double* charge_quadrupole_matrix,
    double* charge_potentials, double* dipole_potentials, double* quadrupole_potentials,
    double* energies, double* gradients, double* strain_derivatives, double* coordination_adjoint,
    std::string& error);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_PERIODIC_MULTIPOLE_HPP
