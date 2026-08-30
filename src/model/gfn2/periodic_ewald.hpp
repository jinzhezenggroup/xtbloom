// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#ifndef XTBLOOM_MODEL_GFN2_PERIODIC_EWALD_HPP
#define XTBLOOM_MODEL_GFN2_PERIODIC_EWALD_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "model/gfn2/es2.hpp"
#include "model/gfn2/periodic_topology.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {

struct PeriodicEwaldPlanData;

/*
 * CPU reference plan for the shell-resolved periodic GFN2 charge operator.
 * The plan owns only immutable lattice/cutoff metadata; all numerical arrays
 * passed to evaluation remain caller-owned.  Keeping this primitive separate
 * from the SCC driver makes the Ewald decomposition independently testable and
 * gives the CUDA backend a precise reference contract to mirror.
 */
class PeriodicEwaldPlan {
 public:
  PeriodicEwaldPlan() noexcept = default;
  PeriodicEwaldPlan(const PeriodicEwaldPlan&) noexcept = default;
  PeriodicEwaldPlan(PeriodicEwaldPlan&&) noexcept = default;
  PeriodicEwaldPlan& operator=(const PeriodicEwaldPlan&) noexcept = default;
  PeriodicEwaldPlan& operator=(PeriodicEwaldPlan&&) noexcept = default;
  ~PeriodicEwaldPlan() = default;

  [[nodiscard]] bool sealed() const noexcept { return data_ != nullptr; }
  [[nodiscard]] std::int64_t batch_size() const noexcept;
  [[nodiscard]] std::int64_t total_atoms() const noexcept;
  [[nodiscard]] std::int64_t total_shells() const noexcept;
  [[nodiscard]] std::int64_t total_matrix_elements() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& atom_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& batch_shell_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& matrix_offsets() const noexcept;
  [[nodiscard]] double alpha(std::int64_t system) const noexcept;
  [[nodiscard]] double direct_cutoff(std::int64_t system) const noexcept;
  [[nodiscard]] double reciprocal_cutoff(std::int64_t system) const noexcept;
  [[nodiscard]] const PeriodicEwaldPlanData* identity() const noexcept { return data_.get(); }
  [[nodiscard]] bool overlaps_storage(const void* data, std::size_t bytes) const noexcept;

 private:
  explicit PeriodicEwaldPlan(std::shared_ptr<const PeriodicEwaldPlanData> data) noexcept
      : data_(std::move(data)) {}

  std::shared_ptr<const PeriodicEwaldPlanData> data_;

  friend xtbloom_status_t make_periodic_ewald_plan(const ES2Plan&, const PeriodicShortRangePlan&,
                                                   PeriodicEwaldPlan&, std::string&);
  friend xtbloom_status_t evaluate_periodic_ewald_cpu(const PeriodicEwaldPlan&,
                                                      const PeriodicShortRangePlan&, const double*,
                                                      const double*, double*, double*, double*,
                                                      double*, double*, std::string&);
};

/*
 * Build alpha and complete rectangular direct/reciprocal image metadata for
 * every lattice in a ragged batch. Alpha and cutoffs follow the pinned tblite
 * binary64 search contract; derivatives hold alpha fixed.
 */
xtbloom_status_t make_periodic_ewald_plan(const ES2Plan& es2,
                                          const PeriodicShortRangePlan& topology,
                                          PeriodicEwaldPlan& plan, std::string& error);

/*
 * Evaluate the shell Ewald Coulomb matrix, shell potentials, 1/2 q^T A q
 * energies, and fixed-q Cartesian/strain derivatives. Outputs are all
 * component-major ragged buffers: matrix and shell potential follow ES2Plan,
 * energies are one per system, gradients are atom-major xyz, and strain is
 * row-major 3x3 per system.  The operation is transactional: outputs are
 * written only after every system has produced finite values.
 */
xtbloom_status_t evaluate_periodic_ewald_cpu(const PeriodicEwaldPlan& plan,
                                             const PeriodicShortRangePlan& topology,
                                             const double* positions, const double* shell_charges,
                                             double* coulomb_matrix, double* shell_potentials,
                                             double* energies, double* gradients,
                                             double* strain_derivatives, std::string& error);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_PERIODIC_EWALD_HPP
